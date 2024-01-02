from src.sdf_ui import *


def font_example():
    with Context((1080, 1080)) as ctx:
        scale = 0.65
        offset_x = 25
        offset_y = 25

        inflation = 7.5

        sdf = glyph_sdf(ctx, "i", scale, offset_x, offset_y, path="fonts/georgia_regular.ttf")

        sdf.save()

        bg = clear_color(ctx, hex_col("#2C2D35"))
        shadow = sdf.shadow(distance=10, inflate=inflation, transparency=1.0)
        colored_glyph = sdf.fill(hex_col("#e9c46a"), (0.0, 0.0, 0.0, 0.0), inflation)

        bg.alpha_overlay(shadow).alpha_overlay(colored_glyph).to_rgb().show()
