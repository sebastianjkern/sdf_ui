"""A looping generative study of an animated, swirling night sky."""

import math
from pathlib import Path

from PIL import Image, ImageOps, ImageSequence

from sdf_ui import Canvas, color, cos, field, param, sdf, sin
from sdf_ui.learning import Parameter

SIZE = (640, 420)
CENTER = (SIZE[0] / 2, SIZE[1] / 2)
OUTPUT_DIR = Path("out/generative_art")


def _village():
    houses = []
    roofs = []
    windows = []
    for x, base_y, width, height in (
        (282, 352, 25, 36),
        (319, 361, 23, 30),
        (354, 348, 29, 43),
        (397, 361, 22, 30),
        (432, 350, 30, 41),
        (476, 362, 23, 29),
        (510, 352, 25, 37),
        (594, 359, 26, 32),
    ):
        body = sdf.rounded_rect(
            (x + width / 2, base_y + height / 2), (width, height), 2
        )
        roof = sdf.triangle(
            (x - 4, base_y + 4),
            (x + width / 2, base_y - 18),
            (x + width + 4, base_y + 4),
        )
        houses.append(body)
        roofs.append(roof)
        for window_x in (x + width * 0.30, x + width * 0.68):
            windows.append(sdf.rect((window_x, base_y + height * 0.58), (4, 8), 1))

    church = sdf.rect((555, 346), (32, 72), 2)
    church_roof = sdf.triangle(
        (536, 310), (555, 282), (574, 310)
    )
    church_spire = sdf.rect((555, 273), (3, 20))
    houses.append(church)
    roofs.extend((church_roof, church_spire))
    windows.extend(
        (
            sdf.rect((555, 340), (6, 15), 2),
            sdf.rect((545, 368), (5, 9), 1),
            sdf.rect((565, 368), (5, 9), 1),
        )
    )

    return (
        sdf.union_all(houses),
        sdf.union_all(roofs),
        sdf.union_all(windows),
    )


def starry_current_scene():
    """Build an original village-and-cypress scene under moving sky currents."""
    phase = param("phase")
    cx, cy = CENTER
    main_center = (475 + 24 * cos(phase), 118 + 16 * sin(phase))

    strength = 9.0 + 1.0 * sin(phase * 2)
    flow = field.vortex(main_center, 210, strength).add(
        field.radial(main_center, 210, 1.0)
    )
    flow = flow.add(
        field.turbulence(
            frequency=0.011,
            amplitude=1.5 + 0.2 * sin(phase * 2),
            octaves=4,
            seed=41,
        )
    )

    sky_base = color.radial_gradient(
        main_center,
        (0.105, 0.22, 0.49, 1.0),
        (0.01, 0.022, 0.09, 1.0),
        inner=24,
        outer=520,
    )
    cloud_palette = color.radial_gradient(
        main_center,
        (0.16, 0.29, 0.56, 1.0),
        (0.025, 0.052, 0.16, 1.0),
        inner=42,
        outer=510,
    )
    cloud_mask = color.perlin_noise().warp(flow, strength=28.0, steps=20)
    sky_clouds = sky_base.mask(cloud_palette, cloud_mask, mode="mix")
    sky = sky_clouds.warp(flow, strength=48.0, steps=28)

    # Repeated short ellipses act like painted brush marks; the animated flow
    # bends each row into broad, interlocking currents.
    brush_marks = sdf.ellipse((cx, cy), (17, 3.2)).repeat_grid(
        columns=28,
        rows=23,
        spacing=(24, 18),
        center=CENTER,
    )
    brushed_sky = brush_marks.warp(flow, strength=62.0, steps=32)
    brush_color = brushed_sky.fill("#356fb6", (0.0, 0.0, 0.0, 0.0))
    highlight_marks = sdf.ellipse((cx + 11, cy + 8), (10, 1.5)).repeat_grid(
        columns=24,
        rows=20,
        spacing=(28, 18),
        center=(cx + 11, cy + 8),
    )
    highlights = highlight_marks.warp(flow, strength=60.0, steps=32).fill(
        "#82c4df", (0.0, 0.0, 0.0, 0.0)
    ).transparency(0.64)
    shadow_marks = sdf.ellipse((cx - 10, cy - 4), (12, 3.8)).repeat_grid(
        columns=15,
        rows=12,
        spacing=(48, 32),
        center=(cx - 10, cy - 4),
    )
    shadows = shadow_marks.warp(flow, strength=58.0, steps=32).fill(
        "#18366f", (0.0, 0.0, 0.0, 0.0)
    ).transparency(0.42)
    cross_marks = sdf.ellipse((cx, cy), (12, 2.5)).repeat_grid(
        columns=18,
        rows=14,
        spacing=(38, 29),
        center=CENTER,
    ).rotate(-0.22, center=CENTER)
    cross_strokes = cross_marks.warp(flow, strength=58.0, steps=32).fill(
        "#4a82ba", (0.0, 0.0, 0.0, 0.0)
    ).transparency(0.72)
    flow_lines = flow.contours(spacing=18, line_width=0.7).transparency(0.2)
    sky_glaze = color.clear((0.012, 0.028, 0.11, 0.24))

    celestial_points = (
        ((483, 110), 22, "#ffe59a"),
        ((291, 83), 14, "#f7cf72"),
        ((367, 218), 11, "#fff0b5"),
        ((570, 185), 10, "#f6d77e"),
        ((223, 202), 8, "#c4edee"),
    )
    glows = color.clear((0.0, 0.0, 0.0, 0.0))
    stars = None
    for center, radius, tint in celestial_points:
        glow = color.radial_gradient(
            center,
            (1.0, 0.69, 0.28, 0.52),
            (0.12, 0.24, 0.52, 0.0),
            inner=1,
            outer=radius * 2.4,
        )
        glows = glows.alpha_overlay(glow)
        star = sdf.star(
            center, radius, radius * 0.34, rays=8, rotation=-math.pi / 2
        ).union(
            sdf.circle(center, radius * 0.30)
        )
        stars = star if stars is None else stars.union(star)
    stars = stars.warp(flow, strength=2.1, steps=2)
    star_layer = stars.fill("#fff1bd", (0.0, 0.0, 0.0, 0.0))

    moon_center = (570, 64)
    moon = sdf.circle(moon_center, 30).subtract(sdf.circle((583, 54), 28))
    moon_layer = moon.fill("#fff1c2", (0.0, 0.0, 0.0, 0.0))

    back_hill = sdf.polygon(
        ((0, 292), (72, 278), (142, 298), (220, 267), (290, 293),
         (374, 270), (452, 295), (528, 266), (594, 287), (640, 261),
         (640, 420), (0, 420))
    )
    middle_hill = sdf.polygon(
        ((0, 335), (82, 309), (154, 326), (232, 300), (307, 325),
         (383, 305), (456, 330), (535, 300), (640, 327), (640, 420), (0, 420))
    )
    foreground_hill = sdf.polygon(
        ((0, 381), (82, 352), (170, 367), (246, 339), (326, 367),
         (413, 345), (501, 371), (574, 344), (640, 359), (640, 420), (0, 420))
    )
    back_layer = back_hill.fill("#1d3c74", (0.0, 0.0, 0.0, 0.0))
    middle_layer = middle_hill.fill("#142d5c", (0.0, 0.0, 0.0, 0.0))
    foreground_layer = foreground_hill.fill("#0b1d3d", (0.0, 0.0, 0.0, 0.0))

    village, roof_shapes, window_shapes = _village()
    village_layer = village.fill("#0b1b36", (0.0, 0.0, 0.0, 0.0))
    roof_layer = roof_shapes.fill("#172d51", (0.0, 0.0, 0.0, 0.0))
    window_layer = window_shapes.fill("#ffd574", (0.0, 0.0, 0.0, 0.0))

    cypress = sdf.polygon(
        (
            (92, 420), (111, 390), (99, 359), (121, 333), (113, 301),
            (137, 274), (128, 244), (150, 217), (145, 187), (166, 160),
            (158, 132), (181, 104), (177, 76), (199, 43), (211, 78),
            (205, 108), (221, 137), (213, 169), (232, 197), (220, 230),
            (241, 258), (226, 290), (244, 320), (224, 350), (241, 383),
            (220, 420),
        )
    )
    cypress_layer = cypress.fill("#07162d", (0.0, 0.0, 0.0, 0.0))
    cypress_glint = sdf.segment((207, 395), (202, 73), 2.0).fill(
        "#1b4961", (0.0, 0.0, 0.0, 0.0)
    )

    scene = sky_base.alpha_overlay(sky).alpha_overlay(sky_glaze).alpha_overlay(flow_lines)
    scene = scene.alpha_overlay(shadows).alpha_overlay(cross_strokes)
    scene = scene.alpha_overlay(brush_color)
    scene = scene.alpha_overlay(highlights)
    scene = scene.alpha_overlay(glows).alpha_overlay(star_layer)
    scene = scene.alpha_overlay(moon_layer)
    scene = scene.alpha_overlay(back_layer).alpha_overlay(middle_layer)
    scene = scene.alpha_overlay(foreground_layer).alpha_overlay(village_layer)
    scene = scene.alpha_overlay(roof_layer)
    scene = scene.alpha_overlay(window_layer).alpha_overlay(cypress_layer)
    return scene.alpha_overlay(cypress_glint).alpha_overlay(
        color.film_grain().transparency(0.025)
    )


def starry_current_animation(
    frames=48, output=OUTPUT_DIR / "starry_current"
):
    """Render a looping animated sky with an original village landscape."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    phase = Parameter(
        "phase",
        0.0,
        animate=lambda progress: progress * 2 * math.pi,
    )
    with Canvas(SIZE) as canvas:
        scene = starry_current_scene()
        preview = output.with_name(output.name + "_preview.png")
        scene.save(
            preview,
            canvas=canvas,
            params={"phase": phase},
            progress=0.31,
        )
        animation = output.with_suffix(".gif")
        canvas.save_animation(
            scene,
            animation,
            frames=frames,
            params={"phase": phase},
            fps=24,
        )

    # GPU textures use a bottom-left origin; export the artwork in the usual
    # top-left image orientation for both the still and animation.
    with Image.open(preview) as image:
        ImageOps.flip(image).save(preview)
    with Image.open(animation) as image:
        duration = image.info.get("duration", round(1000 / 24))
        loop = image.info.get("loop", 0)
        frames_out = [
            ImageOps.flip(frame.convert("RGB")) for frame in ImageSequence.Iterator(image)
        ]
    frames_out[0].save(
        animation,
        save_all=True,
        append_images=frames_out[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )


if __name__ == "__main__":
    starry_current_animation()
