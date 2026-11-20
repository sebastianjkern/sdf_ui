"""Show layered color, refraction, blur, and edge light in liquid glass."""

from pathlib import Path

from sdf_ui import Canvas, color, field, sdf

SIZE = (1080, 1080)
OUTPUT_DIR = Path("out/examples/transparency")


def _background_grid(palette):
    fine = sdf.grid((0, 0), (48, 48)).stroke(
        0.8, (*palette.sample(0.72)[:3], 0.35), (0.0, 0.0, 0.0, 0.0)
    )
    coarse = sdf.grid((0, 0), (192, 192)).stroke(
        1.5, (*palette.sample(0.90)[:3], 0.46), (0.0, 0.0, 0.0, 0.0)
    )
    return fine.alpha_overlay(coarse)


def _background_disc(center, radius, face_color):
    shape = sdf.circle(center, radius)
    disc = shape.light(
        fg_color=face_color, bg_color=(0.0, 0.0, 0.0, 0.0),
        light_dir=(-12 / 25, -29 / 50, 39 / 50), ambient=0.30,
        diffuse=1.0, specular=0.62, shininess=28.0,
        normal_strength=7.0, bevel=radius * 7 / 20,
    )
    highlight = color.radial_gradient(
        (center[0] - radius * 17 / 50, center[1] + radius * 9 / 25),
        (1.0, 1.0, 0.82, 0.30), (*face_color[:3], 0.0),
        inner=0, outer=radius * 23 / 20,
    )
    disc = disc.alpha_overlay(shape.fill_from_texture(highlight))
    shadow = shape.translate((0.0, -7.0)).shadow(distance=16, inflate=0, transparency=0.38)
    return shadow.alpha_overlay(disc)


def _background_discs(width, height, palette):
    circles = color.clear((0.0, 0.0, 0.0, 0.0))
    specs = (
        ((width * 1 / 5, height * 19 / 25), width * 21 / 100, 39 / 50),
        ((width * 79 / 100, height * 3 / 4), width * 1 / 4, 0.96),
        ((width * 39 / 50, height * 23 / 100), width * 1 / 5, 0.66),
        ((width * 1 / 5, height * 1 / 5), width * 17 / 100, 0.88),
    )
    for center, radius, sample in specs:
        circles = circles.alpha_overlay(
            _background_disc(center, radius, palette.sample(sample))
        )
    return circles


def _background(width, height, palette):
    """Build a moss-toned backdrop, regular grid, and foreground color discs."""
    base = color.radial_gradient(
        (width * 23 / 50, height * 47 / 100),
        palette.sample(0.48), palette.sample(0.02), inner=20, outer=width * 39 / 50,
    )
    return base, _background_grid(palette), _background_discs(width, height, palette)


def _panel_geometry(width, height):
    center = (width * 0.5, height * 0.5)
    size = (width * 33 / 100, height * 17 / 50)
    return sdf.rounded_rect(center, size, width * 19 / 200).cache("glass_panel")


def _refracted_pane(base, grid, circles, backdrop, panel, width, height):
    refraction = field.vortex(
        (width * 53 / 100, height * 27 / 50), radius=width * 31 / 50, strength=1.0
    ).add(field.turbulence(frequency=0.009, amplitude=2.4, octaves=3, seed=17))
    pane = base.warp(refraction, strength=15, steps=7).alpha_overlay(grid)
    pane = pane.alpha_overlay(circles.warp(refraction, strength=15, steps=7))
    glass_view = pane.mask(backdrop, panel.generate_mask(), mode="mix")
    frosting = panel.fill_from_texture(pane.post.blur(n=1, base=24)).transparency(0.78)
    return glass_view.alpha_overlay(frosting)


def _glass_finish(panel, width, height):
    transparent = (0.0, 0.0, 0.0, 0.0)
    tint = panel.fill((0.58, 41 / 50, 0.94, 0.13), transparent)
    sheen = color.linear_gradient(
        (width * 20 / 100, height * 41 / 50), (width * 41 / 50, height * 9 / 50),
        (0.93, 0.98, 1.0, 20 / 100), (0.62, 0.84, 0.98, 0.015),
    )
    sheen_layer = panel.fill_from_texture(sheen)
    shadow = panel.shadow(30, 5, transparency=0.48)
    rim = panel.stroke(2.0, (0.93, 0.99, 1.0, 41 / 50), transparent)
    inner_rim = panel.stroke(
        7.0, (0.58, 0.81, 0.94, 0.24), transparent, inflate=-5.0
    )
    return tint, sheen_layer, shadow, rim, inner_rim


def _reflections(width, height):
    broad = sdf.rounded_rect(
        (width * 0.50, height * 161 / 200), (width * 13 / 50, 7.0), 8.0
    ).fill((1.0, 1.0, 1.0, 0.37), (0.0, 0.0, 0.0, 0.0))
    tail = sdf.segment(
        (width * 6 / 25, height * 149 / 200),
        (width * 17 / 50, height * 379 / 500), 1.5,
    ).fill((0.91, 0.98, 1.0, 0.55), (0.0, 0.0, 0.0, 0.0))
    return broad, tail


def transparency_example():
    """Render a glass panel that bends and softens the colorful scene behind it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = SIZE
    with Canvas(SIZE) as canvas:
        base, grid, circles = _background(width, height, color.palette("moss"))
        backdrop = base.alpha_overlay(grid).alpha_overlay(circles)
        panel = _panel_geometry(width, height)
        glass = _refracted_pane(base, grid, circles, backdrop, panel, width, height)
        tint, sheen, shadow, rim, inner_rim = _glass_finish(panel, width, height)
        reflections = _reflections(width, height)
        image = backdrop.alpha_overlay(shadow).alpha_overlay(glass).alpha_overlay(tint)
        for layer in (sheen, inner_rim, rim, *reflections):
            image = image.alpha_overlay(layer)
        image.save(OUTPUT_DIR / "transparency_square.png", canvas=canvas)


if __name__ == "__main__":
    transparency_example()
