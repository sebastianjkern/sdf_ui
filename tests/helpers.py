from pathlib import Path

import numpy as np

from sdf_ui import Canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def render(texture, size=(32, 32), params=None, cache=None):
    ctx = Canvas(size)
    return texture.render(ctx, params=params, cache=cache)


def rgba_array(texture):
    width, height = texture.tex.size
    components = getattr(texture.tex, "components", 4)
    pixels = np.frombuffer(texture.tex.read(), dtype=np.uint8).reshape(
        (height, width, components)
    )
    return np.flipud(pixels)
