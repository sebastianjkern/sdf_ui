from sdf_ui import Canvas, color
from sdf_ui.text import text
from sdf_ui.util import hex_col


def text_rendering_example():
    with Canvas((960*2, 360*2)) as ctx:
        cache = {}
        background = color.clear(hex_col("#20232a")).cache("background")
        title = text(
            "Cached SDF text",
            size=74*2,
            ox=42,
            oy=78,
            path="fonts/georgia_regular.ttf",
            cache_size=160,
        ).cache("title")
        subtitle = text(
            "glyphs are rendered once,\nthen resized into this texture",
            size=38*2,
            ox=46,
            oy=178,
            path="fonts/georgia_regular.ttf",
            cache_size=128,
            line_height=1.25,
        ).cache("subtitle")

        scene = (
            background.alpha_overlay(title.shadow(14, inflate=1.5, transparency=0.65))
            .alpha_overlay(title.fill(hex_col("#e9c46a"), "#00000000", inflate=1.5))
            .alpha_overlay(subtitle.fill(hex_col("#f4f1de"), "#00000000"))
        )
        scene.to_rgb().show(ctx, cache=cache)
