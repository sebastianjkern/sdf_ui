"""Render a 4K geometric forest wallpaper with liquid-glass highlights."""

import argparse
from pathlib import Path

from PIL import Image

from sdf_ui import Canvas, color, cos, field, param, sdf, sin

DEFAULT_SIZE = (3840, 2160)
DEFAULT_OUTPUT = Path("out/wallpapers/liquid_glass_geometric_forest.png")


def geometric_forest_scene(size=DEFAULT_SIZE, animated=False):
    """Create an opaque sky, sculpted glass forest, and liquid reflections."""
    width, height = size
    scale = width / 1920.0
    palette = color.palette("sea_glass")
    horizon = height * 0.76

    if animated:
        phase = param("phase") * (2.0 * 3.141592653589793)
        main_center = (
            width * 0.57 + 24 * scale * cos(phase),
            height * 0.34 + 13 * scale * sin(phase),
        )
        counter_center = (
            width * 0.22 + 12 * scale * cos(phase + 1.3),
            height * 0.49 + 8 * scale * sin(phase + 1.3),
        )
        main_strength = 5.8 + 0.55 * sin(phase)
        counter_strength = -1.7 + 0.22 * cos(phase)
    else:
        main_center = (width * 0.57, height * 0.34)
        counter_center = (width * 0.22, height * 0.49)
        main_strength = 5.8
        counter_strength = -1.7
    flow = field.vortex(
        main_center, radius=width * 0.47, strength=main_strength
    ).add(
        field.vortex(
            counter_center, radius=width * 0.30, strength=counter_strength
        )
    )
    flow = flow.add(
        field.turbulence(
            frequency=0.0032 / scale,
            amplitude=6.5 * scale,
            octaves=4,
            seed=29,
        )
    )

    sky_base = color.radial_gradient(
        (width * 0.50, height * 0.38),
        (0.035, 0.16, 0.28, 1.0),
        (0.006, 0.018, 0.055, 1.0),
        inner=40 * scale,
        outer=width * 0.72,
    )
    sky = sky_base.warp(flow, strength=28 * scale, steps=10)
    scene = sky_base.alpha_overlay(sky)

    # Layered, warped ellipse strokes create glassy aurora ribbons.
    aurora_specs = (
        ((width * 0.47, height * 0.36), (width * 0.40, height * 0.16), -0.12, 11, (0.22, 0.83, 0.76, 0.48)),
        ((width * 0.55, height * 0.32), (width * 0.34, height * 0.12), 0.10, 8, (0.35, 0.80, 0.88, 0.43)),
        ((width * 0.39, height * 0.40), (width * 0.25, height * 0.09), -0.21, 6, (0.18, 0.62, 0.72, 0.38)),
    )
    aurora_glow = color.clear((0.0, 0.0, 0.0, 0.0))
    for ribbon_center, radii, angle, thickness, tint in aurora_specs:
        ribbon = sdf.ellipse(ribbon_center, radii).rotate(
            angle, center=ribbon_center
        )
        warped = ribbon.warp(flow, strength=16 * scale, steps=9)
        aurora_glow = aurora_glow.alpha_overlay(
            warped.stroke(
                thickness * scale,
                tint,
                (0.0, 0.0, 0.0, 0.0),
            ).transparency(0.43)
        )
    scene = scene.alpha_overlay(aurora_glow)

    # A luminous glass moon anchors the upper-right sky.
    orb_center = (width * 0.72, height * 0.29)
    orb_radius = height * 0.105
    orb_rgb = palette.sample(0.88)[:3]
    orb_halo = color.radial_gradient(
        orb_center,
        (*orb_rgb, 0.32),
        (*orb_rgb, 0.0),
        inner=orb_radius * 0.4,
        outer=orb_radius * 2.7,
    )
    orb = sdf.circle(orb_center, orb_radius)
    orb_fill = orb.fill(
        (0.45, 0.82, 0.82, 0.17), (0.0, 0.0, 0.0, 0.0)
    )
    orb_rim = orb.stroke(
        2.8 * scale, (0.73, 0.98, 0.94, 0.82), (0.0, 0.0, 0.0, 0.0)
    )
    orb_inner_rim = sdf.ring(orb_center, orb_radius * 0.82, 1.2 * scale).stroke(
        1.1 * scale, (0.40, 0.82, 0.83, 0.42), (0.0, 0.0, 0.0, 0.0)
    )
    orb_glint = sdf.segment(
        (orb_center[0] - orb_radius * 0.42, orb_center[1] - orb_radius * 0.36),
        (orb_center[0] - orb_radius * 0.06, orb_center[1] - orb_radius * 0.49),
        3.2 * scale,
    ).fill((0.93, 1.0, 0.93, 0.9), (0.0, 0.0, 0.0, 0.0))
    scene = (
        scene.alpha_overlay(orb_halo)
        .alpha_overlay(orb_fill)
        .alpha_overlay(orb_rim)
        .alpha_overlay(orb_inner_rim)
        .alpha_overlay(orb_glint)
    )

    # Tiny lights sit behind the canopy, like distant stars caught in glass.
    stars = sdf.union_all(
        [
            sdf.circle((width * x, height * y), radius * scale)
            for x, y, radius in (
                (0.14, 0.22, 2.2),
                (0.24, 0.13, 1.7),
                (0.34, 0.27, 2.5),
                (0.46, 0.12, 1.8),
                (0.60, 0.19, 2.0),
                (0.84, 0.15, 2.4),
                (0.91, 0.36, 1.6),
                (0.79, 0.45, 1.9),
            )
        ]
    )
    scene = scene.alpha_overlay(
        stars.fill((0.74, 0.96, 0.91, 0.78), (0.0, 0.0, 0.0, 0.0))
    )

    # The lake is a dark reflective plane beneath the tree line.
    water_shape = sdf.rect(
        (width * 0.5, (horizon + height) * 0.5),
        (width * 1.1, height - horizon),
        0,
    )
    water = water_shape.fill(
        (0.012, 0.085, 0.13, 0.94), (0.0, 0.0, 0.0, 0.0)
    )
    scene = scene.alpha_overlay(water)

    far_specs = (
        (0.02, 0.75, 0.22, 0.11),
        (0.13, 0.74, 0.24, 0.11),
        (0.28, 0.74, 0.20, 0.10),
        (0.40, 0.74, 0.16, 0.09),
        (0.60, 0.74, 0.16, 0.09),
        (0.80, 0.74, 0.22, 0.10),
        (0.95, 0.75, 0.23, 0.11),
    )
    far_trees = sdf.union_all(
        [
            sdf.fir_tree(
                width * x, height * base, height * tree_height, width * tree_width, 1
            )
            for x, base, tree_height, tree_width in far_specs
        ]
    )
    far_trees = far_trees.warp(flow, strength=3.0 * scale, steps=3)
    scene = scene.alpha_overlay(
        far_trees.fill((0.045, 0.17, 0.20, 0.88), (0.0, 0.0, 0.0, 0.0))
    )

    mid_specs = (
        (0.05, 0.82, 0.40, 0.20),
        (0.24, 0.81, 0.31, 0.16),
        (0.76, 0.82, 0.31, 0.16),
        (0.95, 0.82, 0.42, 0.21),
    )
    mid_trees = [
        sdf.fir_tree(
            width * x, height * base, height * tree_height, width * tree_width, 3
        )
        for x, base, tree_height, tree_width in mid_specs
    ]
    mid_forest = sdf.union_all(mid_trees).warp(flow, strength=4.0 * scale, steps=4)
    mid_fill = mid_forest.fill(
        (0.018, 0.105, 0.14, 0.92), (0.0, 0.0, 0.0, 0.0)
    )
    mid_edges = mid_forest.stroke(
        1.7 * scale, (0.19, 0.63, 0.64, 0.54), (0.0, 0.0, 0.0, 0.0)
    )
    scene = scene.alpha_overlay(mid_fill).alpha_overlay(mid_edges)

    front_specs = (
        (0.08, 0.94, 0.47, 0.23),
        (0.31, 0.90, 0.30, 0.15),
        (0.69, 0.90, 0.30, 0.15),
        (0.92, 0.94, 0.49, 0.24),
    )
    front_trees = [
        sdf.fir_tree(
            width * x, height * base, height * tree_height, width * tree_width, 4
        )
        for x, base, tree_height, tree_width in front_specs
    ]

    # Fine horizontal capsules catch the aurora on the lake surface.
    water_marks = sdf.ellipse(
        (width * 0.5, height * 0.85), (width * 0.028, 1.3 * scale)
    ).repeat_grid(
        columns=9,
        rows=5,
        spacing=(width * 0.10, height * 0.035),
        center=(width * 0.5, height * 0.85),
    )
    water_marks = water_marks.warp(flow, strength=6.0 * scale, steps=7)
    water_glints = water_marks.fill(
        (0.35, 0.78, 0.80, 0.45), (0.0, 0.0, 0.0, 0.0)
    ).transparency(0.5)
    horizon_glow = sdf.segment(
        (width * 0.12, horizon), (width * 0.88, horizon), 1.3 * scale
    ).fill((0.31, 0.71, 0.72, 0.42), (0.0, 0.0, 0.0, 0.0))
    scene = scene.alpha_overlay(water_glints).alpha_overlay(horizon_glow)

    front_forest = sdf.union_all(front_trees).warp(flow, strength=5.5 * scale, steps=4)
    front_fill = front_forest.fill(
        (0.035, 0.16, 0.19, 0.88), (0.0, 0.0, 0.0, 0.0)
    )
    front_edges = front_forest.stroke(
        2.8 * scale, (0.38, 0.88, 0.84, 0.82), (0.0, 0.0, 0.0, 0.0)
    )
    scene = scene.alpha_overlay(front_fill).alpha_overlay(front_edges)

    # Long, tapered speculars give the nearest glass trees a polished edge.
    speculars = color.clear((0.0, 0.0, 0.0, 0.0))
    for x, base, tree_height, tree_width in (front_specs[0], front_specs[-1]):
        cx = width * x
        base_y = height * base
        tree_h = height * tree_height
        tree_w = width * tree_width
        for side, tint in ((-1, (0.45, 0.87, 0.84, 0.72)), (1, (0.25, 0.64, 0.69, 0.48))):
            highlight = sdf.segment(
                (cx, base_y - tree_h),
                (cx + side * tree_w * 0.43, base_y - tree_h * 0.12),
                1.5 * scale,
            ).fill(tint, (0.0, 0.0, 0.0, 0.0))
            speculars = speculars.alpha_overlay(highlight)
    return scene.alpha_overlay(speculars)


def render_wallpaper(size=DEFAULT_SIZE, output=DEFAULT_OUTPUT):
    """Render the geometric liquid-glass forest to a PNG wallpaper."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Canvas(size) as canvas:
        geometric_forest_scene(size).save(output, canvas=canvas)
    # SDF coordinates use a bottom-left origin; wallpaper layouts use top-left.
    with Image.open(output) as rendered:
        rendered.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=DEFAULT_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_SIZE[1])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.width < 1 or args.height < 1:
        parser.error("width and height must be positive")
    print(render_wallpaper((args.width, args.height), args.output))


if __name__ == "__main__":
    main()
