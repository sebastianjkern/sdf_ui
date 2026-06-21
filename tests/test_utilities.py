from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from sdf_ui import Canvas
from sdf_ui.bw_to_sdf import image_to_sdf
from sdf_ui.core.plugins.primitives.text.plugin import _cached_glyph_sdf
from sdf_ui.text import glyph, text
from sdf_ui.util import hex_col
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


def test_hex_col_supports_short_and_long_alpha_forms():
    assert hex_col("#fff") == (1.0, 1.0, 1.0, 1.0)
    assert hex_col("#fff0") == (1.0, 1.0, 1.0, 0.0)
    assert hex_col("#00000000") == (0.0, 0.0, 0.0, 0.0)
    assert hex_col("#000000", alpha=0) == (0.0, 0.0, 0.0, 0.0)


def test_glyph_renders_using_the_bundled_font():
    font_path = PROJECT_ROOT / "fonts" / "georgia_regular.ttf"
    scene = glyph("i", 0.65, 25, 25, path=str(font_path))

    with Canvas((128, 128)) as ctx:
        texture = scene.fill((255, 255, 255, 255), (0, 0, 0, 255)).render(ctx)

    assert texture.tex.size == (128, 128)
    assert texture.kind == "color"


def test_glyph_generates_a_filled_signed_field_with_counters():
    font_path = PROJECT_ROOT / "fonts" / "georgia_regular.ttf"
    scene = glyph("o", 0.08, 20, 15, path=str(font_path))

    with Canvas((128, 128)) as ctx:
        texture = scene.render(ctx)
        distances = np.frombuffer(texture.tex.read(), dtype=np.float32).reshape(
            (128, 128)
        )

    assert texture.kind == "sdf"
    assert distances[55, 60] > 0
    assert distances[55, 30] < 0
    assert distances[55, 90] < 0


def test_glyph_fill_preserves_transparent_hex_background():
    font_path = PROJECT_ROOT / "fonts" / "georgia_regular.ttf"
    scene = glyph("o", 0.08, 20, 15, path=str(font_path)).fill(
        "#e9c46a", "#00000000"
    )

    with Canvas((128, 128)) as ctx:
        texture = scene.to_rgb().render(ctx)
        pixels = np.frombuffer(texture.tex.read(), dtype=np.uint8).reshape(
            (128, 128, 4)
        )

    assert pixels[55, 30, 3] == 255
    assert pixels[55, 60, 3] == 0


def test_glyph_fill_normalizes_255_tuple_colors():
    font_path = PROJECT_ROOT / "fonts" / "georgia_regular.ttf"
    scene = glyph("o", 0.08, 20, 15, path=str(font_path)).fill(
        (233, 196, 106, 255), (0, 0, 0, 0)
    )

    with Canvas((128, 128)) as ctx:
        texture = scene.to_rgb().render(ctx)
        pixels = np.frombuffer(texture.tex.read(), dtype=np.uint8).reshape(
            (128, 128, 4)
        )

    assert int(pixels[55, 30, 0]) == pytest.approx(233, abs=2)
    assert int(pixels[55, 30, 1]) == pytest.approx(196, abs=2)
    assert int(pixels[55, 30, 2]) == pytest.approx(106, abs=2)
    assert int(pixels[55, 30, 3]) == 255
    assert int(pixels[55, 60, 3]) == 0


def test_text_renders_multiple_glyphs_from_cached_sdf_patches():
    font_path = PROJECT_ROOT / "fonts" / "georgia_regular.ttf"
    _cached_glyph_sdf.cache_clear()
    scene = text("ii", size=72, ox=12, oy=12, path=str(font_path), cache_size=96)

    with Canvas((160, 96)) as ctx:
        texture = scene.render(ctx)
        distances = np.frombuffer(texture.tex.read(), dtype=np.float32).reshape(
            (96, 160)
        )

    cache_info = _cached_glyph_sdf.cache_info()
    assert texture.kind == "sdf"
    assert cache_info.hits == 1
    assert cache_info.misses == 1
    assert distances.min() < 0


def test_sdf_namespace_exposes_text_factory():
    from sdf_ui import sdf

    with Canvas((96, 96)) as ctx:
        texture = sdf.text("A", ox=12, oy=12).render(ctx)
        distances = np.frombuffer(texture.tex.read(), dtype=np.float32).reshape(
            (96, 96)
        )

    assert texture.kind == "sdf"
    assert distances.min() < 0
