"""Text detection: locate text regions and build a mask for inpainting.

The primary detector uses PaddleOCR's DBNet text detector, which returns
line-level quadrilaterals. Lines are grouped into blocks (approximating speech
bubbles / caption boxes) and a dilated binary mask of all text pixels is
produced for the inpainting stage. When PaddleOCR is unavailable a lightweight
OpenCV MSER-based fallback keeps the app functional.

All coordinates are in the input image's pixel space. Callers working on
webtoon tiles offset the results back into full-image space via ``tiling``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectedRegion:
    bbox: list[int]                      # [x, y, w, h]
    polygon: list[list[int]] = field(default_factory=list)
    lines: list[list[int]] = field(default_factory=list)  # member line bboxes
    vertical: bool = False               # source text orientation, if known
    font_size: int = 0                   # estimated pixel font size, if known


class TextDetector:
    """Detects text regions. Backend is chosen lazily on first use.

    Preference order: the vendored comic-text-detector (best for manga/comics,
    also yields a stroke-level inpaint mask and orientation), then PaddleOCR's
    DBNet, then a classical OpenCV MSER fallback.
    """

    def __init__(self, lang: str = "japan", device: str = "auto") -> None:
        self._paddle = None
        self._comic = None
        self._device = device
        self._paddle_lang = _paddle_lang(lang)
        self._backend = "uninitialized"

    # ------------------------------------------------------------- backends
    def _ensure_backend(self) -> None:
        if self._backend != "uninitialized":
            return
        # 1) Vendored comic-text-detector (preferred).
        try:
            from .comic_detector import ComicTextDetector

            self._comic = ComicTextDetector(device=self._device)
            self._comic._ensure_model()
            self._backend = "comic"
            return
        except Exception:  # noqa: BLE001 - fall back to PaddleOCR
            self._comic = None
        # 2) PaddleOCR DBNet detector.
        try:
            from paddleocr import PaddleOCR

            self._paddle = PaddleOCR(
                use_angle_cls=False,
                lang=self._paddle_lang,
                show_log=False,
            )
            self._backend = "paddle"
        except Exception:  # noqa: BLE001 - fall back to classical CV
            self._paddle = None
            self._backend = "mser"

    # -------------------------------------------------------------- detect
    def detect(self, image: np.ndarray) -> tuple[list[DetectedRegion], np.ndarray]:
        """Return grouped regions and a binary text mask (uint8 0/255)."""
        self._ensure_backend()
        if self._backend == "comic" and self._comic is not None:
            try:
                return self._comic.detect(image)
            except Exception:  # noqa: BLE001 - degrade for this call only
                pass
        if self._backend == "paddle":
            polys = self._detect_paddle(image)
        else:
            polys = self._detect_mser(image)
        mask = _mask_from_polygons(image.shape[:2], polys)
        line_boxes = [_poly_to_box(p) for p in polys]
        regions = _group_lines(line_boxes, polys)
        return regions, mask

    def _detect_paddle(self, image: np.ndarray) -> list[list[list[int]]]:
        assert self._paddle is not None
        try:
            result = self._paddle.ocr(image, det=True, rec=False, cls=False)
        except Exception:  # noqa: BLE001
            return []
        polys: list[list[list[int]]] = []
        # PaddleOCR returns a per-image list; det-only yields a list of quads.
        blocks = result[0] if result else None
        if not blocks:
            return []
        for quad in blocks:
            pts = quad[0] if (isinstance(quad, (list, tuple)) and quad and
                              isinstance(quad[0], (list, tuple))
                              and len(quad) == 2) else quad
            try:
                polys.append([[int(x), int(y)] for x, y in pts])
            except (TypeError, ValueError):
                continue
        return polys

    def _detect_mser(self, image: np.ndarray) -> list[list[list[int]]]:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        mser = cv2.MSER_create()
        mser.setMinArea(40)
        mser.setMaxArea(int(0.05 * gray.shape[0] * gray.shape[1]))
        regions, _ = mser.detectRegions(gray)
        boxes = [cv2.boundingRect(r.reshape(-1, 1, 2)) for r in regions]
        # Merge character-level boxes into line-ish clusters via dilation.
        mask = np.zeros(gray.shape, np.uint8)
        for x, y, w, h in boxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys: list[list[list[int]]] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 200:
                continue
            polys.append([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        return polys


# --------------------------------------------------------------------- helpers
def _paddle_lang(lang: str) -> str:
    return {
        "ja": "japan",
        "ko": "korean",
        "zh": "ch",
        "en": "en",
    }.get(lang, "japan")


def _poly_to_box(poly: list[list[int]]) -> list[int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, y0 = min(xs), min(ys)
    return [x0, y0, max(xs) - x0, max(ys) - y0]


def _mask_from_polygons(shape: tuple[int, int],
                        polys: list[list[list[int]]]) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, np.uint8)
    for poly in polys:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    if polys:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def _group_lines(boxes: list[list[int]],
                 polys: list[list[list[int]]]) -> list[DetectedRegion]:
    """Cluster nearby line boxes into blocks approximating bubbles.

    Two lines join the same block when they overlap horizontally and are
    vertically close (within roughly one line height). This is a simple,
    dependency-free union-find over the line boxes.
    """
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(n):
        xi, yi, wi, hi = boxes[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = boxes[j]
            h_ref = max(hi, hj, 1)
            x_overlap = min(xi + wi, xj + wj) - max(xi, xj)
            v_gap = max(yi, yj) - min(yi + hi, yj + hj)
            if x_overlap > -0.5 * min(wi, wj) and v_gap < 0.8 * h_ref:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    regions: list[DetectedRegion] = []
    for members in groups.values():
        xs0 = min(boxes[k][0] for k in members)
        ys0 = min(boxes[k][1] for k in members)
        xs1 = max(boxes[k][0] + boxes[k][2] for k in members)
        ys1 = max(boxes[k][1] + boxes[k][3] for k in members)
        bbox = [xs0, ys0, xs1 - xs0, ys1 - ys0]
        poly = [[xs0, ys0], [xs1, ys0], [xs1, ys1], [xs0, ys1]]
        regions.append(
            DetectedRegion(
                bbox=bbox,
                polygon=poly,
                lines=[boxes[k] for k in sorted(members, key=lambda k: boxes[k][1])],
            )
        )
    # Reading order: top-to-bottom, then right-to-left (manga convention).
    regions.sort(key=lambda r: (r.bbox[1], -r.bbox[0]))
    return regions
