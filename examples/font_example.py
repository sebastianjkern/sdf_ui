from sdf_ui import Canvas, color
from sdf_ui.text import glyph_sdf
from sdf_ui.util import hex_col


def font_example():
    with Canvas((1080, 1080)) as ctx:
        scale = 0.65
        offset_x = 25
        offset_y = 25

        inflation = 7.5

        glyph = glyph_sdf("i", scale, offset_x, offset_y, path="fonts/georgia_regular.ttf").cache("glyph")

        cache = {}

        bg = color.clear(hex_col("#2C2D35")).cache("background")
        shadow = glyph.shadow(distance=10, inflate=inflation, transparency=1.0)
        colored_glyph = glyph.fill(hex_col("#e9c46a"), (0.0, 0.0, 0.0, 0.0), inflation)

        glyph.save("out/font_glyph.png", ctx, cache=cache)
        bg.alpha_overlay(shadow).alpha_overlay(colored_glyph).to_rgb().show(ctx, cache=cache)
