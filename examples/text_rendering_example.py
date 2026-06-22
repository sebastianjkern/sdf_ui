from pathlib import Path

from sdf_ui import Canvas, color
from sdf_ui.text import text
from sdf_ui.util import hex_col

OUTPUT_DIR = Path("out/examples")


def text_rendering_example():
    with Canvas((1080, 1080)) as ctx:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cache = {}
        background = color.clear(hex_col("#20232a")).cache("background")
        title = text(
            "Cached\nSDF text",
            size=96,
            ox=72,
            oy=124,
            path="fonts/georgia_regular.ttf",
            cache_size=160,
        ).cache("title")
        subtitle = text(
            "glyphs are rendered once,\nthen resized into the scene",
            size=42,
            ox=76,
            oy=392,
            path="fonts/georgia_regular.ttf",
            cache_size=128,
            line_height=1.25,
        ).cache("subtitle")

        scene = (
            background.alpha_overlay(title.shadow(14, inflate=1.5, transparency=0.65))
            .alpha_overlay(title.fill(hex_col("#e9c46a"), "#00000000", inflate=1.5))
            .alpha_overlay(subtitle.fill(hex_col("#f4f1de"), "#00000000"))
        )
        with ctx.session(cache=cache) as renderer:
            rendered = renderer.render(scene.to_rgb())
            rendered.save(str(OUTPUT_DIR / "text_square.png"), conversion=False)

            stats = renderer.stats
            cache_info = renderer.cache_info()
            print(
                "Render stats: "
                f"{stats.shader_dispatches} dispatches, "
                f"{stats.texture_allocations} textures, "
                f"{cache_info.hits} cache hits, "
                f"{cache_info.misses} cache misses, "
                f"{stats.elapsed_seconds:.3f}s"
            )
