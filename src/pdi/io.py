"""Validação e serialização de imagens PNG suportadas."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def load_png(source) -> tuple[np.ndarray, str]:
    """Carrega somente PNG L (8 bits) ou RGB (24 bits), sem conversão silenciosa."""
    image = Image.open(source)
    if image.format != "PNG":
        raise ValueError("O arquivo precisa estar no formato PNG.")
    if image.mode not in {"L", "RGB"}:
        raise ValueError(
            f"Modo PNG '{image.mode}' não suportado. Use escala de cinza L (8 bits) "
            "ou RGB (24 bits)."
        )
    return np.asarray(image, dtype=np.uint8), image.mode


def to_png_bytes(array: np.ndarray) -> bytes:
    """Codifica uma matriz uint8 2-D ou RGB em PNG."""
    data = np.clip(array, 0, 255).astype(np.uint8)
    mode = "L" if data.ndim == 2 else "RGB"
    buffer = BytesIO()
    Image.fromarray(data, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()

