"""Anthropic Claude provider using the ``anthropic`` SDK."""
from __future__ import annotations

import base64

from PIL import Image

from .base import Provider, ProviderError, TranslationResult, image_to_bytes, parse_translation_response
from .prompts import (
    translation_system_prompt,
    translation_user_prompt,
    vision_ocr_prompt,
)


class ClaudeProvider(Provider):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-latest",
                 supports_vision: bool = True) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ProviderError("The 'anthropic' package is required for Claude.") from exc
        if not api_key:
            raise ProviderError("Claude requires an API key.")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model or "claude-3-5-sonnet-latest"
        self.supports_vision = supports_vision

    def translate(self, texts: list[str], source_lang: str, target_lang: str,
                  glossary_block: str = "", context: str = "") -> TranslationResult:
        if not texts:
            return TranslationResult()
        system = translation_system_prompt(source_lang, target_lang, glossary_block)
        user = translation_user_prompt(texts, context)
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                temperature=0.2,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Translation request failed: {exc}") from exc
        return parse_translation_response(_text_of(resp), expected=len(texts))

    def vision_ocr(self, image: Image.Image, source_lang: str) -> str:
        b64 = base64.b64encode(image_to_bytes(image)).decode("ascii")
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                temperature=0.0,
                system=vision_ocr_prompt(source_lang),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": "Transcribe the text."},
                        ],
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Vision OCR request failed: {exc}") from exc
        return _text_of(resp).strip()

    def test(self) -> str:
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=5,
                messages=[{"role": "user", "content": "Reply with: OK"}],
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(str(exc)) from exc
        return f"Connected. Model replied: {_text_of(resp).strip() or '(empty)'}"


def _text_of(resp) -> str:
    """Concatenate text blocks from a Claude messages response."""
    try:
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
    except (AttributeError, TypeError):
        return ""
