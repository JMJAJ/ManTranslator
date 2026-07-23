"""Wrapper around the vendored comic-text-detector (DBNet + YOLO block det).

This adapts dmMaze/comic-text-detector (cloned into ``.repos``) to the app's
:class:`~mantranslator.core.detection.DetectedRegion` interface. The model
returns, per page, a refined text mask (ideal for inpainting) and a list of
text blocks that each carry member line polygons, an orientation flag and an
estimated font size.

The model weights are downloaded on first use into the app's models cache. If
the repository or weights are unavailable, importing/using this module raises,
and :mod:`mantranslator.core.detection` falls back to PaddleOCR/MSER.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

from ..config import (
    COMIC_DETECTOR_DIR,
    COMIC_DETECTOR_URL,
    COMIC_DETECTOR_WEIGHTS,
    MODELS_DIR,
)
from .detection import DetectedRegion


class ComicDetectorUnavailable(RuntimeError):
    """Raised when the vendored detector cannot be loaded."""


def _select_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    try:
        import torch

        if device in ("auto", "cuda") and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _ensure_repo_on_path() -> None:
    repo = str(COMIC_DETECTOR_DIR)
    if not COMIC_DETECTOR_DIR.exists():
        raise ComicDetectorUnavailable(
            f"comic-text-detector not found at {repo}. Clone it into .repos."
        )
    if repo not in sys.path:
        # Insert at the front so its top-level modules (basemodel, utils,
        # models) resolve to the vendored copy.
        sys.path.insert(0, repo)


def _ensure_weights(progress=None) -> Path:
    if COMIC_DETECTOR_WEIGHTS.exists():
        return COMIC_DETECTOR_WEIGHTS
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = COMIC_DETECTOR_WEIGHTS.with_suffix(".part")

    def _hook(block_num, block_size, total_size):
        if progress and total_size > 0:
            progress(min(block_num * block_size, total_size), total_size)

    urllib.request.urlretrieve(COMIC_DETECTOR_URL, tmp, _hook)  # noqa: S310
    tmp.replace(COMIC_DETECTOR_WEIGHTS)
    return COMIC_DETECTOR_WEIGHTS


class ComicTextDetector:
    """Lazily-loaded comic-text-detector producing regions + inpaint mask."""

    def __init__(self, device: str = "auto", input_size: int = 1024) -> None:
        self._device = _select_device(device)
        self._input_size = input_size
        self._model = None
        self._refine_mode = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        _ensure_repo_on_path()
        weights = _ensure_weights()
        try:
            from inference import TextDetector as _CTD  # type: ignore
            from utils.textmask import REFINEMASK_INPAINT  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise ComicDetectorUnavailable(
                f"Failed to import comic-text-detector: {exc}"
            ) from exc
        try:
            self._model = _CTD(
                model_path=str(weights),
                input_size=self._input_size,
                device=self._device,
                act="leaky",
            )
        except Exception as exc:  # noqa: BLE001
            raise ComicDetectorUnavailable(
                f"Failed to load comic-text-detector model: {exc}"
            ) from exc
        self._refine_mode = REFINEMASK_INPAINT

    def detect(self, image: np.ndarray) -> tuple[list[DetectedRegion], np.ndarray]:
        self._ensure_model()
        assert self._model is not None
        _, mask_refined, blk_list = self._model(
            image, refine_mode=self._refine_mode, keep_undetected_mask=True
        )
        mask = _to_binary_mask(mask_refined, image.shape[:2])
        regions = [self._region_from_block(blk) for blk in blk_list]
        regions = [r for r in regions if r is not None]
        # Manga reading order: top-to-bottom, right-to-left.
        regions.sort(key=lambda r: (r.bbox[1], -r.bbox[0]))
        return regions, mask

    @staticmethod
    def _region_from_block(blk) -> DetectedRegion | None:
        try:
            x1, y1, x2, y2 = (int(v) for v in blk.xyxy)
        except (AttributeError, TypeError, ValueError):
            return None
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if w == 0 or h == 0:
            return None
        line_boxes: list[list[int]] = []
        for line in getattr(blk, "lines", []) or []:
            pts = np.asarray(line).reshape(-1, 2)
            if pts.size == 0:
                continue
            lx0, ly0 = int(pts[:, 0].min()), int(pts[:, 1].min())
            lx1, ly1 = int(pts[:, 0].max()), int(pts[:, 1].max())
            line_boxes.append([lx0, ly0, lx1 - lx0, ly1 - ly0])
        font_size = int(getattr(blk, "font_size", 0) or 0)
        return DetectedRegion(
            bbox=[x1, y1, w, h],
            polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            lines=line_boxes,
            vertical=bool(getattr(blk, "vertical", False)),
            font_size=font_size,
        )


def _to_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    if mask is None:
        return np.zeros(shape, np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 30).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(binary, kernel, iterations=1)
