"""End-to-end translation pipeline.

Given a :class:`Page`, the pipeline detects text, reads it (OCR with an
optional vision-LLM fallback), batch-translates it through the configured AI
provider while honoring the project glossary, erases the originals with
inpainting, and re-renders the translations in a matched font/color. Results
are written back onto the ``Page``/``TextRegion`` objects and saved as an image.

The pipeline is UI-agnostic: progress and cancellation are delivered through a
small :class:`PipelineHooks` callback object so it can run on a worker thread.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from ..ai.base import Provider, ProviderError
from ..ai.registry import build_translation_provider, build_vision_provider
from ..config import Settings
from ..project.glossary import Glossary, GlossaryEntry
from .detection import DetectedRegion, TextDetector
from .inpaint import Inpainter
from .models import Alignment, Orientation, Page, PageStatus, TextRegion
from .ocr import OcrEngine
from .render import (
    FontLibrary,
    estimate_text_color,
    pick_stroke_color,
    render_region,
)
from .tiling import boxes_overlap, make_tiles, offset_box, offset_polygon

# OCR confidence below this triggers the vision-LLM fallback (if configured).
VISION_FALLBACK_THRESHOLD = 0.5


@dataclass
class PipelineHooks:
    """Optional progress/cancellation callbacks for a running pipeline."""

    on_progress: Callable[[str, int, int], None] | None = None
    is_cancelled: Callable[[], bool] | None = None

    def progress(self, stage: str, current: int, total: int) -> None:
        if self.on_progress:
            self.on_progress(stage, current, total)

    def cancelled(self) -> bool:
        return bool(self.is_cancelled and self.is_cancelled())


class Cancelled(RuntimeError):
    """Raised internally when a hook reports cancellation."""


class TranslationPipeline:
    """Reusable pipeline bound to a set of settings and a glossary."""

    def __init__(self, settings: Settings, glossary: Glossary,
                 source_lang: str, target_lang: str) -> None:
        self.settings = settings
        self.glossary = glossary
        self.source_lang = source_lang
        self.target_lang = target_lang

        self._detector = TextDetector(lang=source_lang, device=settings.device)
        self._ocr = OcrEngine(source_lang=source_lang, engine=settings.ocr_engine)
        self._inpainter = Inpainter(
            use_lama=settings.use_inpainting, device=settings.device
        )
        self._fonts = FontLibrary()
        self._provider: Provider | None = None
        self._vision: Provider | None = None

    # ------------------------------------------------------------- providers
    def _ensure_providers(self) -> None:
        if self._provider is None:
            self._provider = build_translation_provider(self.settings)
        if self._vision is None:
            self._vision = build_vision_provider(self.settings)

    # ------------------------------------------------------------ page entry
    def process_page(self, page: Page, output_path: str | Path,
                     hooks: PipelineHooks | None = None) -> Page:
        hooks = hooks or PipelineHooks()
        try:
            self._ensure_providers()
            image_bgr = _imread(page.image_path)
            if image_bgr is None:
                raise ProviderError(f"Could not read image: {page.image_path}")
            page.width = image_bgr.shape[1]
            page.height = image_bgr.shape[0]

            regions, mask = self._detect(image_bgr, page, hooks)
            self._check(hooks)

            self._ocr_regions(image_bgr, regions, mask, hooks)
            self._check(hooks)

            self._translate_regions(regions, hooks)
            self._check(hooks)

            clean_bgr = self._erase(image_bgr, mask, hooks)
            self._check(hooks)

            out = self._render(clean_bgr, regions, hooks)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out.save(output_path)

            page.regions = regions
            page.output_path = str(output_path)
            page.status = PageStatus.RENDERED.value
            page.error = ""
        except Cancelled:
            page.status = PageStatus.PENDING.value
            raise
        except Exception as exc:  # noqa: BLE001 - surface to UI
            page.status = PageStatus.ERROR.value
            page.error = str(exc)
            raise
        return page

    # -------------------------------------------------------------- rerender
    def rerender_page(self, page: Page, output_path: str | Path) -> Page:
        """Re-draw an already-translated page after manual edits (no AI calls)."""
        image_bgr = _imread(page.image_path)
        if image_bgr is None:
            return page
        mask = self._mask_from_regions(image_bgr.shape[:2], page.regions)
        clean = self._inpainter.inpaint(image_bgr, mask)
        out = self._render(clean, page.regions, PipelineHooks())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.save(output_path)
        page.output_path = str(output_path)
        page.status = PageStatus.RENDERED.value
        return page

    # -------------------------------------------------------------- detection
    def _detect(self, image_bgr: np.ndarray, page: Page,
                hooks: PipelineHooks) -> tuple[list[TextRegion], np.ndarray]:
        hooks.progress("Detecting text", 0, 1)
        h, w = image_bgr.shape[:2]
        full_mask = np.zeros((h, w), np.uint8)
        raw: list[DetectedRegion] = []

        tiles = make_tiles(image_bgr) if page.is_strip else [
            _single_tile(image_bgr)
        ]
        for tile in tiles:
            self._check(hooks)
            det_regions, tile_mask = self._detector.detect(tile.array)
            full_mask[tile.y0:tile.y1] = np.maximum(
                full_mask[tile.y0:tile.y1], tile_mask
            )
            for r in det_regions:
                raw.append(
                    DetectedRegion(
                        bbox=offset_box(r.bbox, tile.y0),
                        polygon=offset_polygon(r.polygon, tile.y0),
                        lines=[offset_box(ln, tile.y0) for ln in r.lines],
                        vertical=r.vertical,
                        font_size=r.font_size,
                    )
                )

        merged = _dedupe_regions(raw)
        stem = Path(page.image_path).stem
        regions: list[TextRegion] = []
        for i, det in enumerate(merged):
            font_size = (
                det.font_size
                or _median_line_height(det.lines)
                or max(12, int(det.bbox[3] * 0.6))
            )
            regions.append(
                TextRegion(
                    id=f"{stem}_{i}",
                    bbox=det.bbox,
                    polygon=det.polygon,
                    reading_order=i,
                    orientation=self._target_orientation(det),
                    font_size=int(font_size),
                    alignment=Alignment.CENTER.value,
                )
            )
        page.status = PageStatus.DETECTED.value
        hooks.progress("Detecting text", 1, 1)
        return regions, full_mask

    # -------------------------------------------------------------------- ocr
    def _ocr_regions(self, image_bgr: np.ndarray, regions: list[TextRegion],
                     mask: np.ndarray, hooks: PipelineHooks) -> None:
        total = len(regions)
        for i, region in enumerate(regions):
            self._check(hooks)
            hooks.progress("Reading text (OCR)", i, total)
            x, y, w, h = region.bbox
            crop = image_bgr[y:y + h, x:x + w]
            crop_mask = mask[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            text, conf = self._ocr.read(crop)
            if conf < VISION_FALLBACK_THRESHOLD and self._vision is not None:
                v_text = self._vision_ocr(crop)
                if v_text:
                    text, conf = v_text, max(conf, 0.6)
                    region.used_vision_fallback = True
            region.source_text = text
            region.ocr_confidence = round(float(conf), 3)
            color = estimate_text_color(crop, crop_mask)
            region.text_color = list(color)
            region.stroke_color = list(pick_stroke_color(color))
        hooks.progress("Reading text (OCR)", total, total)

    def _vision_ocr(self, crop_bgr: np.ndarray) -> str:
        try:
            pil = Image.fromarray(crop_bgr[:, :, ::-1])
            return self._vision.vision_ocr(pil, self.source_lang).strip()  # type: ignore[union-attr]
        except ProviderError:
            return ""

    # ------------------------------------------------------------- translate
    def _translate_regions(self, regions: list[TextRegion],
                           hooks: PipelineHooks) -> None:
        hooks.progress("Translating", 0, 1)
        pending: list[TextRegion] = []
        for region in regions:
            src = region.source_text.strip()
            if not src:
                continue
            memo = self.glossary.memory.get(src)
            if memo:
                region.translated_text = memo
            else:
                pending.append(region)

        if pending:
            texts = [r.source_text.strip() for r in pending]
            glossary_block = self.glossary.as_prompt_block()
            result = self._provider.translate(  # type: ignore[union-attr]
                texts, self.source_lang, self.target_lang, glossary_block
            )
            for region, translation in zip(pending, result.translations):
                region.translated_text = translation.strip()
                self.glossary.remember(region.source_text.strip(), region.translated_text)
            self._absorb_new_terms(result.new_terms)
        hooks.progress("Translating", 1, 1)

    def _absorb_new_terms(self, new_terms: list[dict]) -> None:
        for term in new_terms:
            source = str(term.get("source", "")).strip()
            translation = str(term.get("translation", "")).strip()
            if not source or not translation:
                continue
            entry = GlossaryEntry(
                source=source,
                translation=translation,
                notes=str(term.get("notes", "")).strip(),
            )
            if str(term.get("type", "")).lower().startswith("char"):
                self.glossary.add_character(entry)
            else:
                self.glossary.add_term(entry)

    # ----------------------------------------------------------------- erase
    def _erase(self, image_bgr: np.ndarray, mask: np.ndarray,
               hooks: PipelineHooks) -> np.ndarray:
        hooks.progress("Erasing original text", 0, 1)
        clean = self._inpainter.inpaint(image_bgr, mask)
        hooks.progress("Erasing original text", 1, 1)
        return clean

    # ---------------------------------------------------------------- render
    def _render(self, clean_bgr: np.ndarray, regions: list[TextRegion],
                hooks: PipelineHooks) -> Image.Image:
        image = Image.fromarray(clean_bgr[:, :, ::-1]).convert("RGB")
        total = len(regions)
        for i, region in enumerate(regions):
            hooks.progress("Rendering", i, total)
            if not region.translated_text.strip():
                continue
            font_path = self._resolve_font(region)
            region.font_name = Path(font_path).stem if font_path else "default"
            render_region(image, region, font_path, draw_stroke=True)
        hooks.progress("Rendering", total, total)
        return image

    def _resolve_font(self, region: TextRegion) -> str:
        if region.font_name:
            path = self._fonts.path_for_name(region.font_name)
            if path:
                return path
        return self._fonts.match(region.translated_text)

    # ---------------------------------------------------------------- helpers
    def _target_orientation(self, det: DetectedRegion) -> str:
        # Render vertically only for CJK targets; latin text is horizontal.
        if self.target_lang in {"ja", "zh"}:
            box = det.bbox
            if det.vertical or box[3] > box[2] * 1.5:
                return Orientation.VERTICAL.value
        return Orientation.HORIZONTAL.value

    def _mask_from_regions(self, shape: tuple[int, int],
                           regions: list[TextRegion]) -> np.ndarray:
        mask = np.zeros(shape, np.uint8)
        for r in regions:
            x, y, w, h = r.bbox
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        return mask

    def _check(self, hooks: PipelineHooks) -> None:
        if hooks.cancelled():
            raise Cancelled()


# --------------------------------------------------------------------- helpers
def _imread(path: str) -> np.ndarray | None:
    """Read an image as BGR, tolerating non-ASCII paths on Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except (OSError, ValueError):
        return None


def _single_tile(image: np.ndarray):
    from .tiling import Tile

    return Tile(index=0, y0=0, y1=image.shape[0], array=image)


def _dedupe_regions(regions: list[DetectedRegion]) -> list[DetectedRegion]:
    """Drop near-duplicate regions produced by overlapping tiles."""
    kept: list[DetectedRegion] = []
    for region in regions:
        if any(boxes_overlap(region.bbox, k.bbox, 0.6) for k in kept):
            continue
        kept.append(region)
    kept.sort(key=lambda r: (r.bbox[1], -r.bbox[0]))
    return kept


def _median_line_height(lines: list[list[int]]) -> int:
    if not lines:
        return 0
    heights = sorted(ln[3] for ln in lines)
    return int(heights[len(heights) // 2] * 0.9)
