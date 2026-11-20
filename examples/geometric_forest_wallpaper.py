"""Render a 4K geometric forest wallpaper with liquid-glass highlights."""

import argparse
from pathlib import Path

from PIL import Image

from sdf_ui import Canvas, color, cos, field, param, sdf, sin

DEFAULT_SIZE = (3840, 2160)
DEFAULT_OUTPUT = Path("out/examples/geometric_forest/liquid_glass_geometric_forest.png")


def geometric_forest_scene(size=DEFAULT_SIZE, animated=False):
    """Create an opaque sky, sculpted glass forest, and liquid reflections."""
    width, height = size
    scale = width / 1920.0
    palette = color.palette("sea_glass")
    horizon = height * 19 / 25
    flow, main_center = _forest_flow(width, height, scale, animated)
    scene = _forest_sky(width, height, scale, flow, main_center)
    scene = _forest_orb(width, height, scale, palette, scene)
    scene = _forest_stars(width, height, scale, scene)
    scene = _forest_lake(width, height, horizon, scene)
    scene = _far_forest(width, height, scale, flow, scene)
    scene = _middle_forest(width, height, scale, flow, scene)
    front_trees, front_specs = _front_trees(width, height)
    scene = _water_reflections(width, height, scale, horizon, flow, scene)
    scene = _front_forest(width, height, scale, flow, scene, front_trees)
    return _tree_speculars(width, height, scale, scene, front_specs)


def _forest_flow(width, height, scale, animated):
    main_center, counter_center, main_strength, counter_strength = _flow_parameters(
        width, height, scale, animated
    )
    flow = field.vortex(main_center, radius=width * 47 / 100, strength=main_strength)
    flow = flow.add(
        field.vortex(counter_center, radius=width * 3 / 10, strength=counter_strength)
    )
    flow = flow.add(field.turbulence(
        frequency=2 / 625 / scale, amplitude=13 / 2 * scale, octaves=4, seed=29,
    ))
    return flow, main_center


def _flow_parameters(width, height, scale, animated):
    if animated:
        phase = param("phase") * (2.0 * 3_141_592_653_589_793 / 1_000_000_000_000_000)
        main_center = (
            width * 57 / 100 + 24 * scale * cos(phase),
            height * 17 / 50 + 13 * scale * sin(phase),
        )
        counter_center = (
            width * 11 / 50 + 12 * scale * cos(phase + 13 / 10),
            height * 49 / 100 + 8 * scale * sin(phase + 13 / 10),
        )
        main_strength = 29 / 5 + 11 / 20 * sin(phase)
        counter_strength = -17 / 10 + 11 / 50 * cos(phase)
    else:
        main_center = (width * 57 / 100, height * 17 / 50)
        counter_center = (width * 11 / 50, height * 49 / 100)
        main_strength = 5.8
        counter_strength = -17 / 10
    return main_center, counter_center, main_strength, counter_strength


def _forest_sky(width, height, scale, flow, main_center):
    sky_base = color.radial_gradient(
        (width * 0.50, height * 19 / 50),
        (0.035, 0.16, 0.28, 1.0),
        (0.006, 0.018, 0.055, 1.0),
        inner=40 * scale,
        outer=width * 18 / 25,
    )
    sky = sky_base.warp(flow, strength=28 * scale, steps=10)
    scene = sky_base.alpha_overlay(sky)

    return _paint_aurora_ribbons(width, height, scale, flow, scene)


def _paint_aurora_ribbons(width, height, scale, flow, scene):
    # Layered, warped ellipse strokes create glassy aurora ribbons.
    aurora_specs = (
        ((width * 47 / 100, height * 9 / 25), (width * 2 / 5, height * 4 / 25), -3 / 25, 11, (11 / 50, 0.83, 0.76, 0.48)),
        ((width * 11 / 20, height * 8 / 25), (width * 17 / 50, height * 3 / 25), 0.10, 8, (0.35, 0.80, 0.88, 0.43)),
        ((width * 39 / 100, height * 2 / 5), (width * 1 / 4, height * 9 / 100), -21 / 100, 6, (0.18, 0.62, 18 / 25, 0.38)),
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
    return scene


def _forest_orb(width, height, scale, palette, scene):
    # A luminous glass moon anchors the upper-right sky.
    orb_center = (width * 18 / 25, height * 29 / 100)
    orb_radius = height * 21 / 200
    orb_rgb = palette.sample(0.88)[:3]
    orb_halo = color.radial_gradient(
        orb_center,
        (*orb_rgb, 0.32),
        (*orb_rgb, 0.0),
        inner=orb_radius * 2 / 5,
        outer=orb_radius * 27 / 10,
    )
    orb = sdf.circle(orb_center, orb_radius)
    orb_fill = orb.fill(
        (0.45, 0.82, 0.82, 0.17), (0.0, 0.0, 0.0, 0.0)
    )
    orb_rim = orb.stroke(
        14 / 5 * scale, (0.73, 0.98, 0.94, 0.82), (0.0, 0.0, 0.0, 0.0)
    )
    orb_inner_rim = sdf.ring(orb_center, orb_radius * 41 / 50, 6 / 5 * scale).stroke(
        11 / 10 * scale, (2 / 5, 0.82, 0.83, 0.42), (0.0, 0.0, 0.0, 0.0)
    )
    orb_glint = _orb_glint(orb_center, orb_radius, scale)
    scene = (
        scene.alpha_overlay(orb_halo)
        .alpha_overlay(orb_fill)
        .alpha_overlay(orb_rim)
        .alpha_overlay(orb_inner_rim)
        .alpha_overlay(orb_glint)
    )
    return scene


def _orb_glint(center, radius, scale):
    return sdf.segment(
        (center[0] - radius * 21 / 50, center[1] - radius * 9 / 25),
        (center[0] - radius * 3 / 50, center[1] - radius * 49 / 100),
        16 / 5 * scale,
    ).fill((0.93, 1.0, 0.93, 0.9), (0.0, 0.0, 0.0, 0.0))


def _forest_stars(width, height, scale, scene):
    # Tiny lights sit behind the canopy, like distant stars caught in glass.
    stars = sdf.union_all(
        [
            sdf.circle((width * x, height * y), radius * scale)
            for x, y, radius in (
                (0.14, 11 / 50, 2.2),
                (0.24, 0.13, 17 / 10),
                (17 / 50, 0.27, 2.5),
                (0.46, 3 / 25, 1.8),
                (0.60, 0.19, 2.0),
                (0.84, 0.15, 2.4),
                (0.91, 9 / 25, 1.6),
                (0.79, 0.45, 1.9),
            )
        ]
    )
    scene = scene.alpha_overlay(
        stars.fill((0.74, 0.96, 0.91, 0.78), (0.0, 0.0, 0.0, 0.0))
    )
    return scene


def _forest_lake(width, height, horizon, scene):
    # The lake is a dark reflective plane beneath the tree line.
    water_shape = sdf.rect(
        (width * 0.5, (horizon + height) * 0.5),
        (width * 11 / 10, height - horizon),
        0,
    )
    water = water_shape.fill(
        (0.012, 0.085, 0.13, 0.94), (0.0, 0.0, 0.0, 0.0)
    )
    scene = scene.alpha_overlay(water)
    return scene


def _far_forest(width, height, scale, flow, scene):
    far_specs = (
        (0.02, 0.75, 11 / 50, 0.11),
        (0.13, 0.74, 0.24, 0.11),
        (0.28, 0.74, 0.20, 0.10),
        (2 / 5, 0.74, 0.16, 0.09),
        (0.60, 0.74, 0.16, 0.09),
        (0.80, 0.74, 11 / 50, 0.10),
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
    return scene


def _middle_forest(width, height, scale, flow, scene):
    mid_specs = (
        (0.05, 0.82, 2 / 5, 0.20),
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
        17 / 10 * scale, (0.19, 0.63, 0.64, 0.54), (0.0, 0.0, 0.0, 0.0)
    )
    scene = scene.alpha_overlay(mid_fill).alpha_overlay(mid_edges)
    return scene


def _front_trees(width, height):
    front_specs = (
        (0.08, 0.94, 47 / 100, 0.23),
        (0.31, 0.90, 0.30, 0.15),
        (0.69, 0.90, 0.30, 0.15),
        (0.92, 0.94, 49 / 100, 0.24),
    )
    front_trees = [
        sdf.fir_tree(
            width * x, height * base, height * tree_height, width * tree_width, 4
        )
        for x, base, tree_height, tree_width in front_specs
    ]
    return front_trees, front_specs


def _water_reflections(width, height, scale, horizon, flow, scene):
    # Fine horizontal capsules catch the aurora on the lake surface.
    water_marks = sdf.ellipse(
        (width * 0.5, height * 17 / 20), (width * 7 / 250, 13 / 10 * scale)
    ).repeat_grid(
        columns=9,
        rows=5,
        spacing=(width * 1 / 10, height * 7 / 200),
        center=(width * 0.5, height * 17 / 20),
    )
    water_marks = water_marks.warp(flow, strength=6.0 * scale, steps=7)
    water_glints = water_marks.fill(
        (0.35, 0.78, 0.80, 0.45), (0.0, 0.0, 0.0, 0.0)
    ).transparency(0.5)
    horizon_glow = sdf.segment(
        (width * 3 / 25, horizon), (width * 22 / 25, horizon), 13 / 10 * scale
    ).fill((0.31, 0.71, 18 / 25, 0.42), (0.0, 0.0, 0.0, 0.0))
    scene = scene.alpha_overlay(water_glints).alpha_overlay(horizon_glow)
    return scene


def _front_forest(width, height, scale, flow, scene, front_trees):
    front_forest = sdf.union_all(front_trees).warp(flow, strength=11 / 2 * scale, steps=4)
    front_fill = front_forest.fill(
        (0.035, 0.16, 0.19, 0.88), (0.0, 0.0, 0.0, 0.0)
    )
    front_edges = front_forest.stroke(
        14 / 5 * scale, (0.38, 0.88, 0.84, 0.82), (0.0, 0.0, 0.0, 0.0)
    )
    scene = scene.alpha_overlay(front_fill).alpha_overlay(front_edges)
    return scene


def _tree_speculars(width, height, scale, scene, front_specs):
    # Long, tapered speculars give the nearest glass trees a polished edge.
    speculars = color.clear((0.0, 0.0, 0.0, 0.0))
    for x, base, tree_height, tree_width in (front_specs[0], front_specs[-1]):
        cx = width * x
        base_y = height * base
        tree_h = height * tree_height
        tree_w = width * tree_width
        for side, tint in ((-1, (0.45, 0.87, 0.84, 18 / 25)), (1, (0.25, 0.64, 0.69, 0.48))):
            highlight = sdf.segment(
                (cx, base_y - tree_h),
                (cx + side * tree_w * 43 / 100, base_y - tree_h * 3 / 25),
                3 / 2 * scale,
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
