from pathlib import Path

from sdf_ui import Canvas, color, sdf
from sdf_ui.text import text

WIDTH = 1984
HEIGHT = 794
FONT = "fonts/roboto/Roboto-Regular (3).ttf"

TRANSPARENT = "#00000000"
FIELD_COLOR = "#10253f"
FIELD_COLOR_ACCENT = "#1b3b63"
WARM_HALO_COLOR = "#8e6b2aac"
COOL_HALO_COLOR = "#1c4f8fac"
GRID_COLOR = "#8fa8c6ff"
SPARK_BLUE_COLOR = "#2f79b8"
SPARK_GREEN_COLOR = "#2d7a5a"
SPARK_GOLD_COLOR = "#d6a23a"
TITLE_ACCENT_COLOR = "#d48b2f"
TITLE_TEXT_COLOR = "#f6f7f9"
SUBTITLE_TEXT_COLOR = "#dde4ec"

def _over(base, *layers):
    for layer in layers:
        base = base.alpha_overlay(layer)
    return base


def github_banner_example(output="banner.png"):
    output = Path(output)
    cache = {}

    with Canvas((WIDTH, HEIGHT)) as ctx:
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

        background = field.light(
            fg_color=FIELD_COLOR,
            bg_color=FIELD_COLOR,
            light_dir=(-0.55, -0.72, 0.42),
            ambient=0.62,
            diffuse=0.62,
            specular=0.05,
            shininess=40.0,
            normal_strength=5.2,
            bevel="7%min",
            shade_background=True,
            background_bevel="7%min",
            height_profile="circle_in",
            height_gamma=0.72,
            height=0.5,
        ).cache("field_core")

        warm_halo = (
            color.radial_gradient(
                (WIDTH * 0.76, HEIGHT * 0.30),
                WARM_HALO_COLOR,
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
                COOL_HALO_COLOR,
                TRANSPARENT,
                inner=0,
                outer=1200,
            )
            .transparency(0.15)
            .cache("cool_halo")
        )

        grid = (
            sdf.grid((0, 0), (96, 96), angle=-0.18)
            .outline(GRID_COLOR, TRANSPARENT)
            .transparency(0.15)
            .cache("grid")
        )

        spark_0 = sdf.circle((WIDTH * 0.89, HEIGHT * 0.28), 34).fill(
            SPARK_BLUE_COLOR, TRANSPARENT
        )
        spark_1 = sdf.triangle(
            (WIDTH * 0.52, HEIGHT * 0.23),
            (WIDTH * 0.56, HEIGHT * 0.17),
            (WIDTH * 0.60, HEIGHT * 0.25),
        )

        spark_1_fill = spark_1.fill(SPARK_GREEN_COLOR, TRANSPARENT, inflate=10)
        spark_1_shadow = spark_1.shadow(10, 10, 0.75)

        spark_2 = sdf.rect(
            (WIDTH * 0.92, HEIGHT * 0.66),
            (95, 95),
            (26, 26, 26, 26),
            angle=0.65,
        )

        spark_2_fill = spark_2.fill(SPARK_GOLD_COLOR, TRANSPARENT)
        spark_2_shadow = spark_2.shadow(10, 0, 0.75)

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
            min_render_size=96,
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
            title_sdf.fill(TITLE_ACCENT_COLOR, TRANSPARENT, inflate=1.2),
            ui_sdf.fill(TITLE_TEXT_COLOR, TRANSPARENT, inflate=1.2)
        ).cache("title")
        subtitle = subtitle_sdf.fill(
            SUBTITLE_TEXT_COLOR, TRANSPARENT, inflate=0.6
        ).cache("subtitle")

        scene = _over(
            background,
            warm_halo,
            cool_halo,
            grid,
            spark_0,

            spark_1_shadow,
            spark_1_fill,
            
            spark_2_shadow,
            spark_2_fill,
            title,
            subtitle,
            color.film_grain().transparency(0.025)
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
