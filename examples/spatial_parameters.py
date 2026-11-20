"""Render artwork with texture-driven geometry, color, and transforms."""

from math import pi
from pathlib import Path

from sdf_ui import Canvas, color, field, sdf

OUTPUT = Path("out/generative_art/spatial_parameters")
SIZE = (720, 720)
CENTER = (360, 360)
SQUARE_GRID_SIZE = (700, 1010)


def halftone():
    flow = field.turbulence(frequency=0.006, amplitude=1, octaves=2, seed=31)
    density = flow.channel("x").at_cell()
    radius = density.remap((-3 / 10, 0.53), (0.0, 5.6))
    dots = sdf.circle((0, 0), radius).repeat_grid(60, 60, (12, 12), CENTER)
    shade = flow.channel("y").at_pixel().remap((-3 / 4, 0.75), (0.1, 0.38))
    ink = dots.ink_linear((0.015, shade, 0.9, 1))
    return color.clear("#ffffff").over(ink).finish(tone=0)


def woven_stamps():
    flow = field.turbulence(frequency=0.005, amplitude=1, octaves=2, seed=12)
    # The red output of a texture-driven shader is itself a parameter source.
    direction_map = color.linear_rgba(
        (flow.channel("x").at_pixel() * 0.5 + 0.5, 0, 0, 1)
    )
    direction = direction_map.channel("red").at_cell()
    bend = flow.channel("y").at_cell()
    angle = direction.remap((0.1, 0.9), (-14 / 5, 2.8))
    top = sdf.rect(
        (0, 0),
        (bend.remap((-1, 1), (4, 13)), 2.6),
        corner_radius=direction.remap((0, 1), (0.2, 2.6)),
        angle=angle,
    )
    bead = sdf.circle((-5, 0), bend.remap((-1, 1), (1, 4))).rotate(angle)
    stamp = top.ink_linear(
        (direction.remap((0, 1), (0.08, 0.8)), 0.35, 0.21, 0.9),
    ).over(bead.ink_linear((0.25, 0.055, 0.13, 0.65)))
    weave = stamp.repeat_grid(24, 24, (30, 30), CENTER)
    paint = weave.transparency(
        flow.channel("x").at_pixel().remap((-0.5, 0.7), (0.36, 1))
    )
    return color.linear_rgba((0.006, 0.016, 0.028, 1)).over(paint).finish(tone=0.2)


def ceramic_tiles():
    # Bake this tiny design once. Subsequent cell samples move and resize it.
    tile = sdf.circle(CENTER, 10).subtract(sdf.circle((363, 362), 8))
    stamp = tile.ink_linear((0.95, 0.56, 0.13, 0.95)).over(
        sdf.circle((354, 354), 3).ink_linear((0.12, 0.63, 0.76, 0.9))
    )
    field_map = field.vortex(CENTER, 210, 1).add(
        field.turbulence(frequency=0.008, amplitude=0.4, seed=44)
    )
    twist = field_map.channel("x").at_cell().remap((-13 / 20, 13 / 20), (-3, 3))
    size = field_map.channel("y").at_cell().remap((-13 / 20, 13 / 20), (0.6, 1.3))
    pattern = stamp.repeat_grid(
        24,
        24,
        (30, 30),
        CENTER,
        stamp_center=CENTER,
        rotation=twist,
        scale=size,
        baked=True,
    )
    return color.linear_rgba((0.025, 0.015, 0.06, 1)).over(pattern).finish(tone=0.25)


def drifting_squares():
    width, height = SQUARE_GRID_SIZE
    spacing = 43
    offset, angle = _square_transform(height, spacing)
    square = sdf.rect(offset, (spacing / 2, spacing / 2), angle=angle)
    outline = square.stroke(1.45, "#171613")

    # Interleave four grids to give each outline room to cross into the next
    # square's space. Composite the strokes so intersections stay visible.
    pitch = (2 * spacing, 2 * spacing)
    layers = (
        outline.repeat_grid(6, 11, pitch, (width / 2 + dx, height / 2 + dy))
        for dx in (-spacing / 2, spacing / 2)
        for dy in (-spacing / 2, spacing / 2)
    )
    return color.clear("#fcfbf8").over(*layers)


def _square_transform(height, spacing):
    depth = sdf.half_plane((0, height - 77), (0, -1)).channel().at_cell()
    drift = depth.remap((0, height - 135), (0, 1))
    jitter = field.turbulence(frequency=1 / spacing, amplitude=1, octaves=1, seed=22)
    twist = field.turbulence(frequency=1 / spacing, amplitude=1, octaves=1, seed=56)
    offset = (
        10 * drift * jitter.channel("x").at_cell(),
        11 * drift * jitter.channel("y").at_cell(),
    )
    angle = drift * twist.channel("x").at_cell().remap((-3 / 5, 0.6), (-pi / 4, pi / 4))
    return offset, angle


def render(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Canvas(SIZE) as canvas:
        for build_scene in (halftone, woven_stamps, ceramic_tiles):
            canvas.save(build_scene(), output / (build_scene.__name__ + ".png"))
    with Canvas(SQUARE_GRID_SIZE) as canvas:
        canvas.save(drifting_squares(), output / "drifting_squares.png")
    return output


if __name__ == "__main__":
    print(render())
