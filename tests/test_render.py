"""Rendering: text fits within the region box after auto-fit wrapping."""
import numpy as np
from PIL import Image

from mantranslator.core.models import TextRegion
from mantranslator.core.render import (
    FontLibrary,
    estimate_text_color,
    render_region,
)


def test_render_stays_within_bounds():
    img = Image.new("RGB", (400, 300), "white")
    region = TextRegion(
        id="t0",
        bbox=[50, 40, 200, 120],
        translated_text="This is a fairly long translated sentence to wrap.",
        font_size=40,
        text_color=[0, 0, 0],
        stroke_color=[255, 255, 255],
    )
    font_path = FontLibrary().match(region.translated_text)
    render_region(img, region, font_path, draw_stroke=False)

    # Pixels changed (text drawn) must fall inside the region bbox, allowing a
    # small margin for the stroke/anti-aliasing.
    arr = np.array(img)
    ys, xs = np.where(np.any(arr < 128, axis=2))
    assert xs.size > 0, "expected some text to be drawn"
    x, y, w, h = region.bbox
    assert xs.min() >= x - 4 and xs.max() <= x + w + 4
    assert ys.min() >= y - 4 and ys.max() <= y + h + 4


def test_estimate_text_color_prefers_dark_ink():
    # White crop with black text pixels marked in the mask.
    crop = np.full((20, 20, 3), 255, dtype=np.uint8)  # BGR white
    crop[5:15, 5:15] = (0, 0, 0)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    r, g, b = estimate_text_color(crop, mask)
    assert r < 60 and g < 60 and b < 60


def test_empty_translation_draws_nothing():
    img = Image.new("RGB", (100, 100), "white")
    before = np.array(img).copy()
    region = TextRegion(id="e", bbox=[10, 10, 50, 30], translated_text="  ")
    render_region(img, region, FontLibrary().match(""), draw_stroke=True)
    assert np.array_equal(before, np.array(img))
