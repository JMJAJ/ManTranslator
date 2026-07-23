"""Background reconstruction (inpainting) behind erased text.

The primary backend is LaMa via ``simple-lama-inpainting``, which produces
clean results even when text sits over artwork. When LaMa or torch is not
available, or the user disables it, an OpenCV Telea inpaint is used as a fast
fallback. Both accept an image plus a binary text mask and return the image
with the masked pixels reconstructed.
"""
from __future__ import annotations

import numpy as np


class Inpainter:
    """Lazily-loaded inpainting backend."""

    def __init__(self, use_lama: bool = True, device: str = "auto") -> None:
        self.use_lama = use_lama
        self.device = device
        self._lama = None
        self._backend: str | None = None

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return
        if self.use_lama:
            try:
                from simple_lama_inpainting import SimpleLama

                self._lama = SimpleLama()
                self._backend = "lama"
                return
            except Exception:  # noqa: BLE001 - fall back to OpenCV
                self._lama = None
        self._backend = "opencv"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return ``image`` (BGR uint8) with ``mask`` (0/255) regions filled."""
        if mask is None or not mask.any():
            return image
        self._ensure_backend()
        if self._backend == "lama":
            return self._inpaint_lama(image, mask)
        return self._inpaint_opencv(image, mask)

    def _inpaint_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from PIL import Image as PILImage

        # SimpleLama expects RGB image and an L-mode mask.
        rgb = PILImage.fromarray(image[:, :, ::-1]) if image.ndim == 3 else \
            PILImage.fromarray(image).convert("RGB")
        mask_img = PILImage.fromarray(mask).convert("L")
        try:
            result = self._lama(rgb, mask_img)  # type: ignore[misc]
        except Exception:  # noqa: BLE001 - degrade to OpenCV on runtime failure
            return self._inpaint_opencv(image, mask)
        out = np.array(result.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        # LaMa may pad to a multiple of 8; crop back to the original size.
        h, w = image.shape[:2]
        return np.ascontiguousarray(out[:h, :w])

    @staticmethod
    def _inpaint_opencv(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import cv2

        mask_u8 = (mask > 0).astype(np.uint8) * 255
        return cv2.inpaint(image, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
