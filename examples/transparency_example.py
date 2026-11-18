"""Show layered color, refraction, blur, and edge light in liquid glass."""

from pathlib import Path

from sdf_ui import Canvas, color, field, sdf

SIZE = (1080, 1080)
OUTPUT_DIR = Path("out/examples")


def _background(width, height, palette):
    """Build a moss-toned backdrop, regular grid, and foreground color discs."""
    base = color.radial_gradient(
        (width * 0.46, height * 0.47),
        palette.sample(0.48),
        palette.sample(0.02),
        inner=20,
        outer=width * 0.78,
    )

    grid = (
        sdf.grid((0, 0), (48, 48)).stroke(
            0.8, (*palette.sample(0.72)[:3], 0.35), (0.0, 0.0, 0.0, 0.0)
        )
        .alpha_overlay(
            sdf.grid((0, 0), (192, 192)).stroke(
                1.5, (*palette.sample(0.90)[:3], 0.46), (0.0, 0.0, 0.0, 0.0)
            )
        )
    )
    circles = color.clear((0.0, 0.0, 0.0, 0.0))
    for center, radius, sample in (
        ((width * 0.20, height * 0.76), width * 0.21, 0.78),
        ((width * 0.79, height * 0.75), width * 0.25, 0.96),
        ((width * 0.78, height * 0.23), width * 0.20, 0.66),
        ((width * 0.20, height * 0.20), width * 0.17, 0.88),
    ):
        face_color = palette.sample(sample)
        disc_shape = sdf.circle(center, radius)
        disc = disc_shape.light(
            fg_color=face_color,
            bg_color=(0.0, 0.0, 0.0, 0.0),
            light_dir=(-0.48, -0.58, 0.78),
            ambient=0.30,
            diffuse=1.0,
            specular=0.62,
            shininess=28.0,
            normal_strength=7.0,
            bevel=radius * 0.35,
        )
        highlight = color.radial_gradient(
            (center[0] - radius * 0.34, center[1] + radius * 0.36),
            (1.0, 1.0, 0.82, 0.30),
            (*face_color[:3], 0.0),
            inner=0,
            outer=radius * 1.15,
        )
        disc = disc.alpha_overlay(disc_shape.fill_from_texture(highlight))
        circle_shadow = disc_shape.translate((0.0, -7.0)).shadow(
            distance=16, inflate=0, transparency=0.38
        )
        disc_with_shadow = circle_shadow.alpha_overlay(disc)
        circles = circles.alpha_overlay(
            disc_with_shadow
        )
    return base, grid, circles


def transparency_example():
    """Render a glass panel that bends and softens the colorful scene behind it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = SIZE
    with Canvas(SIZE) as ctx:
        palette = color.palette("moss")
        base, grid, circles = _background(width, height, palette)
        # Circles sit above the grid so its lines disappear behind their faces.
        backdrop = base.alpha_overlay(grid).alpha_overlay(circles)
        center = (width * 0.5, height * 0.5)
        # Rectangle primitives take half extents rather than full dimensions.
        panel_size = (width * 0.33, height * 0.34)
        corner = width * 0.095
        panel_sdf = sdf.rounded_rect(center, panel_size, corner).cache("glass_panel")

        # The field bends color and light inside the pane; the grid stays square.
        refraction = field.vortex(
            (width * 0.53, height * 0.54), radius=width * 0.62, strength=1.0
        ).add(
            field.turbulence(
                frequency=0.009,
                amplitude=2.4,
                octaves=3,
                seed=17,
            )
        )
        # Keep the glass pane geometrically stable; only refract what is behind it.
        glass_sdf = panel_sdf
        panel_mask = glass_sdf.generate_mask()
        bent_base = base.warp(refraction, strength=15, steps=7)
        bent_circles = circles.warp(refraction, strength=15, steps=7)
        pane_backdrop = bent_base.alpha_overlay(grid).alpha_overlay(bent_circles)
        glass_view = pane_backdrop.mask(backdrop, panel_mask, mode="mix")

        # Frosting softens the refracted image without hiding its color and detail.
        soft_backdrop = pane_backdrop.post.blur(n=1, base=24)
        frosted = glass_sdf.fill_from_texture(soft_backdrop).transparency(0.78)
        glass_view = glass_view.alpha_overlay(frosted)

        tint = glass_sdf.fill(
            (0.58, 0.82, 0.94, 0.13), (0.0, 0.0, 0.0, 0.0)
        )
        sheen = color.linear_gradient(
            (width * 0.20, height * 0.82),
            (width * 0.82, height * 0.18),
            (0.93, 0.98, 1.0, 0.20),
            (0.62, 0.84, 0.98, 0.015),
        )
        sheen_layer = glass_sdf.fill_from_texture(sheen)

        shadow = panel_sdf.shadow(30, 5, transparency=0.48)
        rim = glass_sdf.stroke(
            2.0, (0.93, 0.99, 1.0, 0.82), (0.0, 0.0, 0.0, 0.0)
        )
        inner_rim = glass_sdf.stroke(
            7.0, (0.58, 0.81, 0.94, 0.24), (0.0, 0.0, 0.0, 0.0), inflate=-5.0
        )

        # Broad curved reflections make the surface read as polished glass.
        reflection = sdf.rounded_rect(
            (width * 0.50, height * 0.805),
            (width * 0.26, 7.0),
            8.0,
        ).fill((1.0, 1.0, 1.0, 0.37), (0.0, 0.0, 0.0, 0.0))
        reflection_tail = sdf.segment(
            (width * 0.24, height * 0.745),
            (width * 0.34, height * 0.758),
            1.5,
        ).fill((0.91, 0.98, 1.0, 0.55), (0.0, 0.0, 0.0, 0.0))

        image = (
            backdrop.alpha_overlay(shadow)
            .alpha_overlay(glass_view)
            .alpha_overlay(tint)
            .alpha_overlay(sheen_layer)
            .alpha_overlay(inner_rim)
            .alpha_overlay(rim)
            .alpha_overlay(reflection)
            .alpha_overlay(reflection_tail)
        )
        image.save(OUTPUT_DIR / "transparency_square.png", canvas=ctx)


if __name__ == "__main__":
    transparency_example()
