from pathlib import Path

from sdf_ui import Canvas, color, sdf
from sdf_ui.fonts import default_font_path
from sdf_ui.text import text

WIDTH = 1984
HEIGHT = 794

FONT = default_font_path()

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


def _banner_field():
    blobs = (
        sdf.circle((WIDTH * 3 / 5, HEIGHT * 12 / 25), 136),
        sdf.circle((WIDTH * 37 / 50, HEIGHT * 47 / 100), 176),
        sdf.circle((WIDTH * 67 / 100, HEIGHT * 16 / 25), 118),
    )
    rounded = sdf.rounded_rect(
        (WIDTH * 4 / 5, HEIGHT * 3 / 5), (250, 150), (42, 42, 42, 42), angle=0.18
    )
    return blobs[0].smooth_union(blobs[1], k=42).smooth_union(blobs[2], k=0.5).smooth_union(
        rounded, k=0.5
    ).cache("field")


def _banner_background(field):
    return field.light(
        fg_color=FIELD_COLOR, bg_color=FIELD_COLOR,
        light_dir=(-11 / 20, -18 / 25, 0.42), ambient=0.62, diffuse=0.62,
        specular=0.05, shininess=40.0, normal_strength=5.2, bevel="7%min",
        shade_background=True, background_bevel="7%min", height_profile="circle_in",
        height_gamma=18 / 25, height=0.5,
    ).cache("field_core")


def _banner_halos():
    warm = color.radial_gradient(
        (WIDTH * 19 / 25, HEIGHT * 3 / 10), WARM_HALO_COLOR, TRANSPARENT,
        inner=0, outer=1500,
    ).transparency(0.15).cache("warm_halo")
    cool = color.radial_gradient(
        (WIDTH * 13 / 50, HEIGHT * 18 / 25), COOL_HALO_COLOR, TRANSPARENT,
        inner=0, outer=1200,
    ).transparency(0.15).cache("cool_halo")
    grid = sdf.grid((0, 0), (96, 96), angle=-9 / 50).outline(
        GRID_COLOR, TRANSPARENT
    ).transparency(0.15).cache("grid")
    return warm, cool, grid


def _banner_sparks():
    blue = sdf.circle((WIDTH * 89 / 100, HEIGHT * 7 / 25), 34).fill(
        SPARK_BLUE_COLOR, TRANSPARENT
    )
    green = sdf.triangle(
        (WIDTH * 13 / 25, HEIGHT * 23 / 100),
        (WIDTH * 14 / 25, HEIGHT * 17 / 100),
        (WIDTH * 3 / 5, HEIGHT / 4),
    )
    gold = sdf.rect(
        (WIDTH * 23 / 25, HEIGHT * 33 / 50), (95, 95), (26, 26, 26, 26), angle=0.65
    )
    return (
        blue,
        green.shadow(10, 10, 0.75), green.fill(SPARK_GREEN_COLOR, TRANSPARENT, inflate=10),
        gold.shadow(10, 0, 0.75), gold.fill(SPARK_GOLD_COLOR, TRANSPARENT),
    )


def _banner_title():
    title = text("SDF", size=164, ox=104, oy=410, path=FONT, cache_size=192).cache("title_sdf")
    subtitle = text(
        "A 2D RENDERING ENGINE\nBUILT ON SIGNED DISTANCE FIELDS",
        size=48, ox=112, oy=265, path=FONT, cache_size=192,
        line_height=1.25, oversample=4.0, min_render_size=96,
    ).cache("subtitle_sdf")
    ui = text("UI", size=164, ox=458, oy=410, path=FONT, cache_size=192).cache("ui_sdf")
    heading = color.clear(TRANSPARENT).over(
        title.fill(TITLE_ACCENT_COLOR, TRANSPARENT, inflate=1.2),
        ui.fill(TITLE_TEXT_COLOR, TRANSPARENT, inflate=1.2),
    ).cache("title")
    caption = subtitle.fill(SUBTITLE_TEXT_COLOR, TRANSPARENT, inflate=3 / 5).cache("subtitle")
    return heading, caption


def _banner_scene():
    halos = _banner_halos()
    sparks = _banner_sparks()
    title = _banner_title()
    return _banner_background(_banner_field()).over(
        *halos, *sparks, *title
    ).film_grain().to_rgb()


def github_banner_example(output="out/banner.png"):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Canvas((WIDTH, HEIGHT)) as ctx:
        with ctx.session(cache={}) as renderer:
            rendered = renderer.render(_banner_scene())
            rendered.save(str(output))
            stats, cache = renderer.stats, renderer.cache_info()
            print(
                f"Saved {output} ({stats.shader_dispatches} dispatches, "
                f"{stats.texture_allocations} textures, {cache.hits} cache hits, "
                f"{cache.misses} cache misses, {stats.elapsed_seconds:.3f}s)"
            )
    return output


if __name__ == "__main__":
    github_banner_example()
