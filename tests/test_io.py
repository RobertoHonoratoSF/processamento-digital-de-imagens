from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from pdi.io import load_png, to_png_bytes


def encoded(mode):
    channels = {"RGB": 3, "RGBA": 4}
    shape = (4, 5) if mode == "L" else (4, 5, channels[mode])
    buffer = BytesIO()
    Image.fromarray(np.zeros(shape, dtype=np.uint8), mode=mode).save(buffer, "PNG")
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_png_round_trip(mode):
    image, detected = load_png(encoded(mode))
    assert detected == mode
    restored, restored_mode = load_png(BytesIO(to_png_bytes(image)))
    assert restored_mode == mode
    assert np.array_equal(image, restored)


def test_rejects_rgba():
    with pytest.raises(ValueError, match="não suportado"):
        load_png(encoded("RGBA"))

