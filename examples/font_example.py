from sdf_ui import Canvas, color
from sdf_ui.text import glyph
from sdf_ui.util import hex_col


def font_example():
    with Canvas((1080, 1080)) as ctx:
        scale = 0.65
        offset_x = 25
        offset_y = 25

        inflation = 7.5

        glyph_shape = glyph(
            "ä", scale, offset_x, offset_y, path="fonts/georgia_regular.ttf"
        ).cache("glyph")

        cache = {}

        bg = color.clear(hex_col("#2C2D35")).cache("background")
        shadow = glyph_shape.shadow(distance=10, inflate=inflation, transparency=1.0)
        colored_glyph = glyph_shape.fill(
            hex_col("#e9c46a"), (0.0, 0.0, 0.0, 0.0), inflation
        )

        glyph_shape.save("font_glyph.png", ctx, cache=cache)
        bg.alpha_overlay(shadow).alpha_overlay(colored_glyph).to_rgb().show(
            ctx, cache=cache
        )
