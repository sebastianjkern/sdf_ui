"""Draw a vector field from separate direction and magnitude textures."""

import argparse
from math import pi, tau
from pathlib import Path

import numpy as np
from PIL import Image

from sdf_ui import Canvas, color, sdf

SIZE = (768, 576)
CENTER = (384, 288)
OUTPUT = Path("out/examples/texture_sampled_arrows")
ANGLE_X_FREQUENCY = 0.83
ANGLE_Y_FREQUENCY = 0.61
ANGLE_CROSS_FREQUENCY = 0.94
ANGLE_CROSS_X = 0.27
ANGLE_PRIMARY_AMPLITUDE = 0.29
ANGLE_SECONDARY_AMPLITUDE = 0.16
FLOW_BASELINE = 0.26
FLOW_RAMP = 0.39
FLOW_WAVE = 0.16
FLOW_WAVE_WIDTH = 0.17


def arrow_field(direction_map, magnitude_map):
    """Sample each map at the arrow's center and transform one shared stamp."""
    # Red stores a normalized number. Read it directly without sRGB decoding.
    # A middle gray direction points right, the stamp's default orientation.
    rotation = direction_map.channel("red").at_cell().remap((0, 1), (-pi, pi))
    scale = magnitude_map.channel("red").at_cell().remap((0, 1), (0.40, 1.15))

    cx, cy = CENTER
    stamp = sdf.arrow(
        (cx - 11, cy), (cx + 11, cy), width=2.4, head_length=7, head_width=8
    ).ink("#c7f6ed")
    arrows = stamp.repeat_grid(
        24,
        18,
        (30, 30),
        CENTER,
        stamp_center=CENTER,
        rotation=rotation,
        scale=scale,
        baked=True,
    )
    return color.clear("#0c1c29").over(arrows)


def write_maps(output):
    """Write two example maps. The PNG rows run from top to bottom."""
    width, height = SIZE
    x = np.linspace(0, 1, width)[None, :]
    y = np.linspace(1, 0, height)[:, None]

    # A continuous angle ramp avoids interpolating across a full-turn seam.
    direction = (
        0.5
        + ANGLE_PRIMARY_AMPLITUDE
        * np.sin(tau * (ANGLE_X_FREQUENCY * x + ANGLE_Y_FREQUENCY * y))
        + ANGLE_SECONDARY_AMPLITUDE
        * np.cos(tau * (ANGLE_CROSS_FREQUENCY * y - ANGLE_CROSS_X * x))
    )
    # A separate map gives the current a wide band of stronger flow.
    ridge = y - (FLOW_BASELINE + FLOW_RAMP * x + FLOW_WAVE * np.sin(tau * x))
    magnitude = np.exp(-0.5 * (ridge / FLOW_WAVE_WIDTH) ** 2)

    direction_path = output / "direction.png"
    magnitude_path = output / "magnitude.png"
    for path, values in ((direction_path, direction), (magnitude_path, magnitude)):
        Image.fromarray(np.rint(255 * values).astype(np.uint8)).save(path)
    return direction_path, magnitude_path


def render(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    direction_path, magnitude_path = write_maps(output)
    result = output / "arrow_field.png"
    with (
        Canvas(SIZE) as canvas,
        canvas.load_image(direction_path) as direction_map,
        canvas.load_image(magnitude_path) as magnitude_map,
    ):
        canvas.save(arrow_field(direction_map, magnitude_map), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Output directory")
    print(render(parser.parse_args().output))


if __name__ == "__main__":
    main()
