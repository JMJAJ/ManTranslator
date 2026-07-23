"""OCR engines with per-language selection and a vision-LLM fallback.

Engine selection:

* Japanese  -> manga-ocr (purpose-built for vertical manga text)
* Korean/Chinese/English -> PaddleOCR recognizer
* anything else / fallback -> Tesseract

Each engine returns ``(text, confidence)``. The pipeline decides when a low
confidence or empty result should trigger the optional vision-LLM fallback.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


class OcrEngine:
    """Lazily-initialized multi-backend OCR reader."""

    def __init__(self, source_lang: str = "ja", engine: str = "auto") -> None:
        self.source_lang = source_lang
        self.engine = engine  # auto | manga-ocr | paddleocr | tesseract
        self._manga_ocr = None
        self._paddle = None
        self._resolved: str | None = None

    # --------------------------------------------------------------- resolve
    def _resolve_engine(self) -> str:
        if self._resolved:
            return self._resolved
        choice = self.engine
        if choice == "auto":
            choice = {
                "ja": "manga-ocr",
                "ko": "paddleocr",
                "zh": "paddleocr",
                "en": "paddleocr",
            }.get(self.source_lang, "tesseract")
        self._resolved = choice
        return choice

    # ------------------------------------------------------------------ read
    def read(self, crop: Image.Image | np.ndarray) -> tuple[str, float]:
        """OCR a single text-region crop, returning ``(text, confidence)``."""
        image = _to_pil(crop)
        engine = self._resolve_engine()
        try:
            if engine == "manga-ocr":
                return self._read_manga(image)
            if engine == "paddleocr":
                return self._read_paddle(image)
            return self._read_tesseract(image)
        except Exception:  # noqa: BLE001 - degrade instead of crashing
            # Last-ditch attempt with tesseract, else empty.
            try:
                return self._read_tesseract(image)
            except Exception:  # noqa: BLE001
                return "", 0.0

    # --------------------------------------------------------------- engines
    def _read_manga(self, image: Image.Image) -> tuple[str, float]:
        if self._manga_ocr is None:
            from manga_ocr import MangaOcr

            self._manga_ocr = MangaOcr()
        text = self._manga_ocr(image)
        # manga-ocr does not expose a confidence; empty text means low quality.
        return text.strip(), (0.9 if text.strip() else 0.0)

    def _read_paddle(self, image: Image.Image) -> tuple[str, float]:
        if self._paddle is None:
            from paddleocr import PaddleOCR

            self._paddle = PaddleOCR(
                use_angle_cls=True,
                lang=_paddle_lang(self.source_lang),
                show_log=False,
            )
        arr = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        result = self._paddle.ocr(arr, det=True, rec=True, cls=True)
        lines = result[0] if result else None
        if not lines:
            return "", 0.0
        texts: list[str] = []
        confs: list[float] = []
        for entry in lines:
            try:
                txt, conf = entry[1][0], float(entry[1][1])
            except (IndexError, TypeError, ValueError):
                continue
            texts.append(txt)
            confs.append(conf)
        text = "\n".join(texts).strip()
        conf = float(np.mean(confs)) if confs else 0.0
        return text, conf

    def _read_tesseract(self, image: Image.Image) -> tuple[str, float]:
        import pytesseract

        lang = _tesseract_lang(self.source_lang)
        data = pytesseract.image_to_data(
            image, lang=lang, output_type=pytesseract.Output.DICT
        )
        words = [w for w in data.get("text", []) if w.strip()]
        confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit()]
        confs = [c for c in confs if c >= 0]
        text = " ".join(words).strip()
        conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        return text, conf


# --------------------------------------------------------------------- helpers
def _to_pil(crop: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(crop, Image.Image):
        return crop.convert("RGB")
    arr = crop
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]  # assume BGR from OpenCV -> RGB
    return Image.fromarray(arr).convert("RGB")


def _paddle_lang(lang: str) -> str:
    return {"ja": "japan", "ko": "korean", "zh": "ch", "en": "en"}.get(lang, "en")


def _tesseract_lang(lang: str) -> str:
    return {
        "ja": "jpn",
        "ko": "kor",
        "zh": "chi_sim",
        "en": "eng",
    }.get(lang, "eng")
