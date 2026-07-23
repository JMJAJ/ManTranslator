"""Google Gemini provider using the ``google-generativeai`` SDK."""
from __future__ import annotations

from PIL import Image

from .base import Provider, ProviderError, TranslationResult, parse_translation_response
from .prompts import (
    translation_system_prompt,
    translation_user_prompt,
    vision_ocr_prompt,
)


class GeminiProvider(Provider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash",
                 supports_vision: bool = True) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ProviderError(
                "The 'google-generativeai' package is required for Gemini."
            ) from exc
        if not api_key:
            raise ProviderError("Gemini requires an API key.")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = model or "gemini-1.5-flash"
        # Gemini multimodal models handle vision natively.
        self.supports_vision = supports_vision

    def _model_for(self, system_instruction: str | None = None):
        return self._genai.GenerativeModel(
            self._model, system_instruction=system_instruction
        )

    def translate(self, texts: list[str], source_lang: str, target_lang: str,
                  glossary_block: str = "", context: str = "") -> TranslationResult:
        if not texts:
            return TranslationResult()
        system = translation_system_prompt(source_lang, target_lang, glossary_block)
        user = translation_user_prompt(texts, context)
        try:
            model = self._model_for(system)
            resp = model.generate_content(
                user,
                generation_config={"temperature": 0.2},
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Translation request failed: {exc}") from exc
        return parse_translation_response(_text_of(resp), expected=len(texts))

    def vision_ocr(self, image: Image.Image, source_lang: str) -> str:
        try:
            model = self._model_for(vision_ocr_prompt(source_lang))
            resp = model.generate_content(
                ["Transcribe the text.", image.convert("RGB")],
                generation_config={"temperature": 0.0},
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Vision OCR request failed: {exc}") from exc
        return _text_of(resp).strip()

    def test(self) -> str:
        try:
            model = self._model_for()
            resp = model.generate_content("Reply with: OK")
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(str(exc)) from exc
        return f"Connected. Model replied: {_text_of(resp).strip() or '(empty)'}"


def _text_of(resp) -> str:
    """Extract plain text from a Gemini response defensively."""
    text = getattr(resp, "text", None)
    if text:
        return text
    try:
        parts = resp.candidates[0].content.parts
        return "".join(getattr(p, "text", "") for p in parts)
    except (AttributeError, IndexError):
        return ""
