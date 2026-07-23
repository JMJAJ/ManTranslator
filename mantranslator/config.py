"""Application configuration: settings persistence and well-known paths.

Settings are stored as JSON in the per-user config directory so they survive
between sessions. Model weights and other large caches live under a separate
cache directory that is git-ignored.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


APP_DIR_NAME = "ManTranslator"


def _user_config_dir() -> Path:
    """Return the per-user config directory for the app (cross-platform)."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP_DIR_NAME


def _user_cache_dir() -> Path:
    """Return the per-user cache directory (model weights, etc.)."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / APP_DIR_NAME


CONFIG_DIR = _user_config_dir()
CACHE_DIR = _user_cache_dir()
MODELS_DIR = CACHE_DIR / "models"
SETTINGS_PATH = CONFIG_DIR / "settings.json"

# Bundled asset locations.
PACKAGE_DIR = Path(__file__).resolve().parent
FONTS_DIR = PACKAGE_DIR / "assets" / "fonts"

# Vendored upstream repositories (cloned into <workspace>/.repos).
REPOS_DIR = PACKAGE_DIR.parent / ".repos"
COMIC_DETECTOR_DIR = REPOS_DIR / "comic-text-detector"
# Downloaded model weights for the vendored comic-text-detector.
COMIC_DETECTOR_WEIGHTS = MODELS_DIR / "comictextdetector.pt"
COMIC_DETECTOR_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/"
    "beta-0.2.1/comictextdetector.pt"
)


@dataclass
class ProviderConfig:
    """Connection details for a single AI backend."""

    name: str = "New Provider"
    kind: str = "openai_compat"  # openai_compat | gemini | claude
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = ""
    supports_vision: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderConfig":
        known = {k: data[k] for k in cls.__annotations__ if k in data}
        return cls(**known)


@dataclass
class Settings:
    """Top-level, project-independent application settings."""

    providers: list[ProviderConfig] = field(default_factory=list)
    translation_provider: str = ""   # provider name used for translation
    vision_provider: str = ""        # provider name used for vision-OCR fallback
    source_lang: str = "ja"          # ja | ko | zh | en | ...
    target_lang: str = "en"
    ocr_engine: str = "auto"         # auto | manga-ocr | paddleocr | tesseract
    use_inpainting: bool = True
    device: str = "auto"             # auto | cpu | cuda
    last_project: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["providers"] = [p.to_dict() for p in self.providers]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        providers = [ProviderConfig.from_dict(p) for p in data.get("providers", [])]
        known = {k: data[k] for k in cls.__annotations__ if k in data and k != "providers"}
        return cls(providers=providers, **known)

    def provider(self, name: str) -> ProviderConfig | None:
        for p in self.providers:
            if p.name == name:
                return p
        return None


def ensure_dirs() -> None:
    """Create the config, cache and model directories if they do not exist."""
    for path in (CONFIG_DIR, CACHE_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Load settings from disk, returning defaults when absent or invalid."""
    ensure_dirs()
    if not SETTINGS_PATH.exists():
        return Settings()
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as fh:
            return Settings.from_dict(json.load(fh))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return Settings()


def save_settings(settings: Settings) -> None:
    """Persist settings to disk as pretty-printed JSON."""
    ensure_dirs()
    with SETTINGS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(settings.to_dict(), fh, indent=2, ensure_ascii=False)
