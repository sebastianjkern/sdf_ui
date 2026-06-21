from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sdf_ui import Canvas
from sdf_ui.bw_to_sdf import image_to_sdf
from sdf_ui.text import glyph_sdf

from tests.helpers import PROJECT_ROOT, render, rgba_array


def test_image_to_sdf_renders_a_simple_bi_level_image(tmp_path):
    image_path = tmp_path / "input.png"
    image = Image.new("RGBA", (8, 8), (255, 255, 255, 255))
    for x in range(2, 6):
        for y in range(2, 6):
            image.putpixel((x, y), (0, 0, 0, 255))
    image.save(image_path)

    scene = image_to_sdf(str(image_path), resize=False, preview=False)
    pixels = rgba_array(
        render(scene.fill((255, 255, 255, 255), (0, 0, 0, 255)).to_rgb(), size=(8, 8))
    )

    assert pixels[4, 4, 0] > 200
    assert pixels[0, 0, 0] < 20


def test_image_to_sdf_rejects_images_without_dark_pixels(tmp_path):
    image_path = tmp_path / "blank.png"
    Image.new("RGBA", (8, 8), (255, 255, 255, 255)).save(image_path)

    with pytest.raises(ValueError, match="No dark pixels found"):
        image_to_sdf(str(image_path), resize=False, preview=False)


def test_glyph_sdf_renders_using_the_bundled_font():
    font_path = PROJECT_ROOT / "fonts" / "georgia_regular.ttf"
    scene = glyph_sdf("i", 0.65, 25, 25, path=str(font_path))

    with Canvas((128, 128)) as ctx:
        texture = scene.fill((255, 255, 255, 255), (0, 0, 0, 255)).render(ctx)

    assert texture.tex.size == (128, 128)
    assert texture.kind == "color"
