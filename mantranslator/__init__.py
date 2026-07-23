"""ManTranslator - AI-powered manga/webtoon translator.

Detects text in comic images, OCRs it, translates via pluggable local/API AI
backends, erases originals with LaMa inpainting, and re-renders translations
in a matched font/color while keeping a per-project markdown glossary for
consistent names and terms across chapters.
"""
import os

# Skip PaddleOCR/PaddleX's slow "model hosters" connectivity probe on startup.
# Models still download on first use; this only removes the mirror-speed check.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

__version__ = "0.1.0"
__app_name__ = "ManTranslator"
