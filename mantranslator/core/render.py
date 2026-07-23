"""Text rendering: font matching, auto-fit wrapping, vertical text and color.

``FontLibrary`` catalogs the curated fonts bundled with the app plus any
directories the user adds, and picks the closest match for a region based on
simple style hints (CJK vs latin, emphasis/all-caps). ``render_region`` draws
translated text into a region's bounding box, shrinking and wrapping until it
fits, and supports top-to-bottom vertical layout for Japanese sources.

Color estimation samples the original text pixels (via the detection mask) so
the replacement keeps the source's foreground color.
"""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import FONTS_DIR
from .models import Alignment, Orientation, TextRegion


@dataclass
class FontInfo:
    path: str
    name: str
    is_cjk: bool = False
    is_bold: bool = False


class FontLibrary:
    """Discovers fonts and resolves the best match for a region."""

    def __init__(self, extra_dirs: list[str] | None = None) -> None:
        self._fonts: list[FontInfo] = []
        self._dirs = [str(FONTS_DIR)] + list(extra_dirs or [])
        self.refresh()

    def refresh(self) -> None:
        self._fonts = []
        for directory in self._dirs:
            for pattern in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
                for path in glob.glob(os.path.join(directory, pattern)):
                    self._fonts.append(_font_info(path))

    @property
    def names(self) -> list[str]:
        return [f.name for f in self._fonts]

    def path_for_name(self, name: str) -> str | None:
        for f in self._fonts:
            if f.name == name:
                return f.path
        return None

    def match(self, text: str, prefer_bold: bool = False,
              want_cjk: bool | None = None) -> str:
        """Return a font path best matching the text/style, or a fallback."""
        if want_cjk is None:
            want_cjk = _has_cjk(text)
        candidates = self._fonts
        if want_cjk:
            cjk = [f for f in candidates if f.is_cjk]
            if cjk:
                candidates = cjk
        if prefer_bold:
            bold = [f for f in candidates if f.is_bold]
            if bold:
                candidates = bold
        if candidates:
            return candidates[0].path
        return _system_fallback_font(want_cjk)


@lru_cache(maxsize=256)
def _load_font(path: str, size: int):
    """Load a truetype font at ``size``; fall back to PIL's default font."""
    if path:
        try:
            return ImageFont.truetype(path, size=max(size, 6))
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=max(size, 6))
    except TypeError:  # older Pillow without size arg
        return ImageFont.load_default()


def estimate_text_color(crop_bgr: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    """Estimate the original text (foreground) color as an RGB tuple.

    Uses the mask to select text pixels; if the mask is empty it falls back to
    the darkest cluster in the crop (text is usually darker than the bubble).
    """
    if crop_bgr.ndim != 3:
        return (0, 0, 0)
    rgb = crop_bgr[:, :, ::-1]
    if mask is not None and mask.any():
        pixels = rgb[mask > 0]
    else:
        pixels = rgb.reshape(-1, 3)
    if pixels.size == 0:
        return (0, 0, 0)
    # Prefer the darker half of the masked pixels (ink over anti-aliasing).
    luma = pixels @ np.array([0.299, 0.587, 0.114])
    threshold = np.median(luma)
    dark = pixels[luma <= threshold]
    sample = dark if dark.size else pixels
    color = sample.mean(axis=0)
    return (int(color[0]), int(color[1]), int(color[2]))


def pick_stroke_color(text_color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Choose a contrasting outline color for readability over artwork."""
    luma = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
    return (255, 255, 255) if luma < 128 else (0, 0, 0)


def render_region(image: Image.Image, region: TextRegion, font_path: str,
                  draw_stroke: bool = True) -> None:
    """Draw ``region.translated_text`` into the region's bbox on ``image``."""
    text = region.translated_text.strip()
    if not text:
        return
    draw = ImageDraw.Draw(image)
    x, y, w, h = region.bbox
    if w <= 0 or h <= 0:
        return
    color = tuple(region.text_color)
    stroke = tuple(region.stroke_color)
    stroke_w = max(1, int(round(_best_font_size(region) * 0.06))) if draw_stroke else 0

    if region.orientation == Orientation.VERTICAL.value:
        _render_vertical(draw, text, region, font_path, color, stroke, stroke_w)
    else:
        _render_horizontal(draw, text, region, font_path, color, stroke, stroke_w)


# ------------------------------------------------------------------ horizontal
def _render_horizontal(draw, text, region, font_path, color, stroke, stroke_w) -> None:
    x, y, w, h = region.bbox
    size, lines = _fit_horizontal(draw, text, font_path, w, h,
                                  start_size=_best_font_size(region))
    font = _load_font(font_path, size)
    line_h = _line_height(font)
    total_h = line_h * len(lines)
    cy = y + max(0, (h - total_h) // 2)
    for line in lines:
        lw = draw.textlength(line, font=font)
        cx = _align_x(region.alignment, x, w, lw)
        draw.text((cx, cy), line, font=font, fill=color,
                  stroke_width=stroke_w, stroke_fill=stroke)
        cy += line_h


def _fit_horizontal(draw, text, font_path, w, h, start_size):
    """Shrink font and wrap until the text fits within ``w`` x ``h``."""
    size = max(8, start_size)
    while size >= 8:
        font = _load_font(font_path, size)
        lines = _wrap_text(draw, text, font, w)
        line_h = _line_height(font)
        fits_w = all(draw.textlength(ln, font=font) <= w for ln in lines)
        fits_h = line_h * len(lines) <= h
        if fits_w and fits_h:
            return size, lines
        size -= 1
    font = _load_font(font_path, 8)
    return 8, _wrap_text(draw, text, font, w)


def _wrap_text(draw, text, font, max_width) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if draw.textlength(trial, font=font) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    # For scripts without spaces, fall back to hard character wrapping.
    wrapped: list[str] = []
    for line in lines:
        if draw.textlength(line, font=font) <= max_width:
            wrapped.append(line)
        else:
            wrapped.extend(_hard_wrap(draw, line, font, max_width))
    return wrapped or [""]


def _hard_wrap(draw, line, font, max_width) -> list[str]:
    out, current = [], ""
    for ch in line:
        if draw.textlength(current + ch, font=font) <= max_width or not current:
            current += ch
        else:
            out.append(current)
            current = ch
    if current:
        out.append(current)
    return out


# -------------------------------------------------------------------- vertical
def _render_vertical(draw, text, region, font_path, color, stroke, stroke_w) -> None:
    """Render columns right-to-left, characters top-to-bottom (manga style)."""
    x, y, w, h = region.bbox
    clean = text.replace("\n", "")
    size = max(8, _best_font_size(region))
    while size >= 8:
        font = _load_font(font_path, size)
        char_h = _line_height(font)
        per_col = max(1, h // char_h)
        cols = (len(clean) + per_col - 1) // per_col
        col_w = int(size * 1.1)
        if cols * col_w <= w or size == 8:
            break
        size -= 1
    font = _load_font(font_path, size)
    char_h = _line_height(font)
    per_col = max(1, h // char_h)
    col_w = int(size * 1.1)
    cx = x + w - col_w  # start at the right edge
    idx = 0
    while idx < len(clean) and cx >= x:
        cy = y
        for _ in range(per_col):
            if idx >= len(clean):
                break
            ch = clean[idx]
            cw = draw.textlength(ch, font=font)
            draw.text((cx + (col_w - cw) / 2, cy), ch, font=font, fill=color,
                      stroke_width=stroke_w, stroke_fill=stroke)
            cy += char_h
            idx += 1
        cx -= col_w


# --------------------------------------------------------------------- helpers
def _best_font_size(region: TextRegion) -> int:
    if region.font_size and region.font_size > 0:
        return region.font_size
    # Default: about 70% of the region height, capped for very tall boxes.
    return max(12, min(64, int(region.bbox[3] * 0.7)))


def _line_height(font: ImageFont.FreeTypeFont) -> int:
    ascent, descent = font.getmetrics()
    return int((ascent + descent) * 1.15)


def _align_x(alignment: str, x: int, w: int, line_w: float) -> int:
    if alignment == Alignment.LEFT.value:
        return x
    if alignment == Alignment.RIGHT.value:
        return int(x + w - line_w)
    return int(x + (w - line_w) / 2)


def _font_info(path: str) -> FontInfo:
    name = Path(path).stem
    lower = name.lower()
    is_bold = "bold" in lower or "black" in lower or "heavy" in lower
    is_cjk = any(tag in lower for tag in ("jp", "kr", "sc", "tc", "cjk", "noto", "gothic", "mincho", "hei", "song"))
    return FontInfo(path=path, name=name, is_cjk=is_cjk, is_bold=is_bold)


def _has_cjk(text: str) -> bool:
    for ch in text:
        o = ord(ch)
        if (0x3040 <= o <= 0x30FF) or (0x3400 <= o <= 0x9FFF) or (0xAC00 <= o <= 0xD7A3):
            return True
    return False


def _system_fallback_font(want_cjk: bool) -> str:
    """Locate a reasonable system font when no curated fonts are present."""
    candidates: list[str] = []
    if sys.platform.startswith("win"):
        win = os.environ.get("WINDIR", "C:/Windows")
        base = os.path.join(win, "Fonts")
        candidates = [
            os.path.join(base, "msgothic.ttc" if want_cjk else "arial.ttf"),
            os.path.join(base, "YuGothM.ttc"),
            os.path.join(base, "arial.ttf"),
            os.path.join(base, "segoeui.ttf"),
        ]
    elif sys.platform == "darwin":
        candidates = [
            "/System/Library/Fonts/Hiragino Sans GB.ttc" if want_cjk else "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc" if want_cjk else
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    # Last resort: bundle-free default that PIL always provides (bitmap only).
    return ""


def load_font_or_default(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Public helper: load a truetype font, or PIL's default if unavailable."""
    if path:
        try:
            return _load_font(path, size)
        except OSError:
            pass
    return ImageFont.load_default()
