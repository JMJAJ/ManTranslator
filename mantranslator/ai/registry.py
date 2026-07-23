"""Factory that builds :class:`Provider` instances from ``ProviderConfig``."""
from __future__ import annotations

from ..config import ProviderConfig, Settings
from .base import Provider, ProviderError


def build_provider(cfg: ProviderConfig) -> Provider:
    """Instantiate a concrete provider from its configuration."""
    kind = (cfg.kind or "").lower()
    if kind == "openai_compat":
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            supports_vision=cfg.supports_vision,
        )
    if kind == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(
            api_key=cfg.api_key,
            model=cfg.model or "gemini-1.5-flash",
            supports_vision=cfg.supports_vision or True,
        )
    if kind == "claude":
        from .claude import ClaudeProvider

        return ClaudeProvider(
            api_key=cfg.api_key,
            model=cfg.model or "claude-3-5-sonnet-latest",
            supports_vision=cfg.supports_vision or True,
        )
    raise ProviderError(f"Unknown provider kind: {cfg.kind!r}")


def build_translation_provider(settings: Settings) -> Provider:
    """Build the provider selected for translation."""
    cfg = settings.provider(settings.translation_provider)
    if cfg is None:
        raise ProviderError(
            "No translation provider selected. Configure one in Settings."
        )
    return build_provider(cfg)


def build_vision_provider(settings: Settings) -> Provider | None:
    """Build the vision-OCR fallback provider, if one is selected."""
    if not settings.vision_provider:
        return None
    cfg = settings.provider(settings.vision_provider)
    if cfg is None:
        return None
    provider = build_provider(cfg)
    return provider if provider.supports_vision else None
