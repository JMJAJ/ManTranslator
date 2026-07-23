"""Tiling helpers for very tall webtoon strips.

Detection and inpainting models expect roughly page-sized inputs, so a long
vertical strip is split into overlapping horizontal tiles. Detections are run
per tile and then mapped back to full-image coordinates; regions that straddle
a tile boundary are de-duplicated by the caller using their bounding boxes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tile:
    """A horizontal slice of a tall image."""

    index: int
    y0: int          # top of the tile in full-image coordinates
    y1: int          # bottom (exclusive)
    array: np.ndarray  # the cropped pixels (H, W, C)

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def make_tiles(image: np.ndarray, tile_height: int = 1600,
               overlap: int = 200) -> list[Tile]:
    """Split ``image`` (H, W, C) into overlapping horizontal tiles.

    Short images (shorter than ``tile_height``) return a single tile covering
    the whole image. ``overlap`` gives detectors context across cut lines so
    text near a boundary is not truncated.
    """
    h = image.shape[0]
    if h <= tile_height:
        return [Tile(index=0, y0=0, y1=h, array=image)]

    step = max(tile_height - overlap, 1)
    tiles: list[Tile] = []
    y = 0
    idx = 0
    while y < h:
        y1 = min(y + tile_height, h)
        tiles.append(Tile(index=idx, y0=y, y1=y1, array=image[y:y1]))
        if y1 >= h:
            break
        y += step
        idx += 1
    return tiles


def offset_box(box: list[int], y_offset: int) -> list[int]:
    """Shift an ``[x, y, w, h]`` box down by ``y_offset``."""
    x, y, w, h = box
    return [x, y + y_offset, w, h]


def offset_polygon(polygon: list[list[int]], y_offset: int) -> list[list[int]]:
    return [[int(px), int(py) + y_offset] for px, py in polygon]


def boxes_overlap(a: list[int], b: list[int], iou_threshold: float = 0.5) -> bool:
    """Return True when two ``[x, y, w, h]`` boxes overlap beyond a threshold."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return False
    union = aw * ah + bw * bh - inter
    return union > 0 and (inter / union) >= iou_threshold
