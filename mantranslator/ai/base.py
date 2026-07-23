"""Abstract AI provider interface and shared helpers.

A provider offers up to three capabilities:

* ``translate`` - batch-translate a list of source strings, returning the
  translations plus any newly detected names/terms.
* ``vision_ocr`` - read text out of an image crop (used as an OCR fallback).
* ``test`` - a cheap round-trip used by the GUI's "Test Connection" button.

Concrete providers live in sibling modules and are created via ``registry``.
"""
from __future__ import annotations

import base64
import io
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class TranslationResult:
    """Structured result of a batch translation request."""

    translations: list[str] = field(default_factory=list)
    new_terms: list[dict] = field(default_factory=list)


class ProviderError(RuntimeError):
    """Raised when a provider request fails or returns an unusable response."""


class Provider(ABC):
    """Common interface implemented by every AI backend."""

    #: Whether this provider instance can process images.
    supports_vision: bool = False

    @abstractmethod
    def translate(self, texts: list[str], source_lang: str, target_lang: str,
                  glossary_block: str = "", context: str = "") -> TranslationResult:
        ...

    def vision_ocr(self, image: Image.Image, source_lang: str) -> str:
        """Read text from an image crop. Override in vision-capable providers."""
        raise ProviderError("This provider does not support vision OCR.")

    @abstractmethod
    def test(self) -> str:
        """Return a short status string, or raise :class:`ProviderError`."""
        ...


# --------------------------------------------------------------------- helpers
def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 ``data:`` URL for chat vision APIs."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


def parse_translation_response(raw: str, expected: int) -> TranslationResult:
    """Parse a model's translation reply into a :class:`TranslationResult`.

    The parser is tolerant: it accepts a bare JSON object, JSON wrapped in
    markdown code fences, or - as a last resort - a plain newline-separated
    list. It always returns exactly ``expected`` translation slots so callers
    can zip them back onto their regions safely.
    """
    result = TranslationResult()
    data = _extract_json_object(raw)
    if isinstance(data, dict) and isinstance(data.get("translations"), list):
        result.translations = [str(t) for t in data["translations"]]
        terms = data.get("new_terms")
        if isinstance(terms, list):
            result.new_terms = [t for t in terms if isinstance(t, dict)]
    else:
        # Fallback: treat non-empty lines as sequential translations.
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        result.translations = lines

    # Normalize length to the number of requested items.
    if len(result.translations) < expected:
        result.translations += [""] * (expected - len(result.translations))
    elif len(result.translations) > expected:
        result.translations = result.translations[:expected]
    return result


def _extract_json_object(raw: str):
    """Best-effort extraction of a JSON object from a model response."""
    if not raw:
        return None
    text = raw.strip()
    # Strip markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Grab the outermost {...} span and try again.
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None
