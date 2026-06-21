from pathlib import Path

from sdf_ui import Canvas, color, sdf
from sdf_ui.text import text

WIDTH = 1984
HEIGHT = 794
FONT = "fonts/roboto/Roboto-Regular (3).ttf"

TRANSPARENT = "#00000000"

def _over(base, *layers):
    for layer in layers:
        base = base.alpha_overlay(layer)
    return base


def github_banner_example(output="banner.png"):
    output = Path(output)
    cache = {}

    with Canvas((WIDTH, HEIGHT)) as ctx:
        background = color.linear_gradient(
            (0, 0),
            (WIDTH, HEIGHT),
            "#03152C",
            "#021225",
        ).cache("background")

        warm_halo = (
            color.radial_gradient(
                (WIDTH * 0.76, HEIGHT * 0.30),
                "#bd9c5fac",
                TRANSPARENT,
                inner=0,
                outer=1500,
            )
            .transparency(0.15)
            .cache("warm_halo")
        )
        cool_halo = (
            color.radial_gradient(
                (WIDTH * 0.26, HEIGHT * 0.72),
                "#1f71ebac",
                TRANSPARENT,
                inner=0,
                outer=1200,
            )
            .transparency(0.15)
            .cache("cool_halo")
        )

        grid = (
            sdf.grid((0, 0), (96, 96), angle=-0.18)
            .outline("#ffffffff", TRANSPARENT)
            .transparency(0.15)
            .cache("grid")
        )

        left_blob = sdf.circle((WIDTH * 0.60, HEIGHT * 0.48), 136)
        right_blob = sdf.circle((WIDTH * 0.74, HEIGHT * 0.47), 176)
        lower_blob = sdf.circle((WIDTH * 0.67, HEIGHT * 0.64), 118)
        field = (
            left_blob.smooth_union(right_blob, k=42)
            .smooth_union(lower_blob, k=0.5)
            .smooth_union(
                sdf.rounded_rect(
                    (WIDTH * 0.80, HEIGHT * 0.60),
                    (250, 150),
                    (42, 42, 42, 42),
                    angle=0.18,
                ),
                k=0.5,
            )
            .cache("field")
        )
        field_core = field.fill_from_texture(
            color.linear_gradient(
                (WIDTH * 0.50, HEIGHT * 0.25),
                (WIDTH * 0.88, HEIGHT * 0.73),
                "#c32c1b",
                "#ee8a2d",
            ),
            background=TRANSPARENT,
            inflate=1.0,
        ).cache("field_core")

        spark_0 = sdf.circle((WIDTH * 0.89, HEIGHT * 0.28), 34).fill(
            "#268ac2", TRANSPARENT
        )
        spark_1 = sdf.triangle(
            (WIDTH * 0.52, HEIGHT * 0.23),
            (WIDTH * 0.56, HEIGHT * 0.17),
            (WIDTH * 0.60, HEIGHT * 0.25),
        ).fill("#2c8956", TRANSPARENT)
        spark_2 = sdf.rect(
            (WIDTH * 0.92, HEIGHT * 0.66),
            (95, 95),
            (26, 26, 26, 26),
            angle=0.65,
        ).fill("#f4b82f", TRANSPARENT)

        title_sdf = text(
            "SDF",
            size=164,
            ox=104,
            oy=410,
            path=FONT,
            cache_size=192,
        ).cache("title_sdf")
        subtitle_sdf = text(
            "A 2D RENDERING ENGINE\nBUILT ON SIGNED DISTANCE FIELDS",
            size=48,
            ox=112,
            oy=265,
            path=FONT,
            cache_size=192,
            line_height=1.25,
            oversample=4.0,
        ).cache("subtitle_sdf")

        ui_sdf = text(
            "UI",
            size=164,
            ox=458,
            oy=410,
            path=FONT,
            cache_size=192,
        ).cache("ui_sdf")

        title = _over(
            color.clear(TRANSPARENT),
            title_sdf.fill(
                "#ee8a2d", TRANSPARENT, inflate=1.2
            ),
            ui_sdf.fill("#ffffff", TRANSPARENT, inflate=1.2)
        ).cache("title")
        subtitle = subtitle_sdf.fill("#f3fbf7", TRANSPARENT, inflate=0.6).cache(
            "subtitle"
        )

        scene = _over(
            background,
            warm_halo,
            cool_halo,
            grid,
            field_core,
            # contours,
            spark_0.transparency(1),
            spark_1.transparency(1),
            spark_2.transparency(1),
            title,
            subtitle,
        ).to_rgb()

        with ctx.session(cache=cache) as renderer:
            rendered = renderer.render(scene)
            rendered.save(str(output))
            stats = renderer.stats
            cache_info = renderer.cache_info()
            print(
                f"Saved {output} "
                f"({stats.shader_dispatches} dispatches, "
                f"{stats.texture_allocations} textures, "
                f"{cache_info.hits} cache hits, "
                f"{cache_info.misses} cache misses, "
                f"{stats.elapsed_seconds:.3f}s)"
            )


if __name__ == "__main__":
    github_banner_example()
