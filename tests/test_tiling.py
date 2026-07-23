"""Tiling split/merge and overlap-dedup tests."""
import numpy as np

from mantranslator.core.tiling import boxes_overlap, make_tiles


def test_single_tile_for_short_image():
    img = np.zeros((800, 600, 3), dtype=np.uint8)
    tiles = make_tiles(img, tile_height=1600, overlap=200)
    assert len(tiles) == 1
    assert tiles[0].y0 == 0 and tiles[0].y1 == 800


def test_tiles_cover_full_height_with_overlap():
    h = 5000
    img = np.zeros((h, 400, 3), dtype=np.uint8)
    tiles = make_tiles(img, tile_height=1600, overlap=200)
    assert len(tiles) > 1
    # Tiles start at 0 and the last tile reaches the bottom edge.
    assert tiles[0].y0 == 0
    assert tiles[-1].y1 == h
    # Consecutive tiles overlap by the configured amount (except the last cut).
    for a, b in zip(tiles, tiles[1:]):
        assert b.y0 < a.y1  # overlap exists


def test_boxes_overlap_threshold():
    a = [0, 0, 100, 100]
    assert boxes_overlap(a, [10, 10, 100, 100], 0.5) is True
    assert boxes_overlap(a, [90, 90, 100, 100], 0.5) is False
    assert boxes_overlap(a, [500, 500, 20, 20], 0.5) is False
