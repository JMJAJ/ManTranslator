"""OpenAI-compatible provider.

Covers OpenAI/ChatGPT, DeepSeek, and any local server exposing the OpenAI
``/v1/chat/completions`` API such as LM Studio (``http://localhost:1234/v1``)
and Ollama (``http://localhost:11434/v1``). Vision OCR is only attempted when
the configured model is flagged as vision-capable.
"""
from __future__ import annotations

from PIL import Image

from .base import (
    Provider,
    ProviderError,
    TranslationResult,
    image_to_data_url,
    parse_translation_response,
)
from .prompts import (
    translation_system_prompt,
    translation_user_prompt,
    vision_ocr_prompt,
)


class OpenAICompatProvider(Provider):
    def __init__(self, base_url: str, api_key: str, model: str,
                 supports_vision: bool = False, timeout: float = 60.0) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ProviderError(
                "The 'openai' package is required for OpenAI-compatible providers."
            ) from exc
        # Local servers often accept any non-empty key. A bounded timeout and a
        # single retry keep a wrong/unreachable endpoint from hanging the app.
        self._client = OpenAI(
            base_url=base_url or None,
            api_key=api_key or "not-needed",
            timeout=timeout,
            max_retries=1,
        )
        self._model = model
        self.supports_vision = supports_vision

    def translate(self, texts: list[str], source_lang: str, target_lang: str,
                  glossary_block: str = "", context: str = "") -> TranslationResult:
        if not texts:
            return TranslationResult()
        system = translation_system_prompt(source_lang, target_lang, glossary_block)
        user = translation_user_prompt(texts, context)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001 - normalize SDK errors
            raise ProviderError(f"Translation request failed: {exc}") from exc
        content = resp.choices[0].message.content or ""
        return parse_translation_response(content, expected=len(texts))

    def vision_ocr(self, image: Image.Image, source_lang: str) -> str:
        if not self.supports_vision:
            raise ProviderError("This model is not configured for vision.")
        data_url = image_to_data_url(image)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": vision_ocr_prompt(source_lang)},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe the text."},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Vision OCR request failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()

    def test(self) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Reply with: OK"}],
                max_tokens=5,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(str(exc)) from exc
        text = (resp.choices[0].message.content or "").strip()
        return f"Connected. Model replied: {text or '(empty)'}"
