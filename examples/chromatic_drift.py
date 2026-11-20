"""Print a seeded diamond lattice that breaks apart from navy to rust."""

import argparse
from contextlib import contextmanager
from math import pi, sqrt
from pathlib import Path

import numpy as np

from sdf_ui import Canvas, ColorTexture, color, sdf

SIZE = (768, 1024)
LANDSCAPE_SIZE = (768, 576)
COLUMNS = 34
PITCH = 20
LEFT, BOTTOM = 44, 88
TOP = 48
PRINT_WIDTH = COLUMNS * PITCH
OUTPUT = Path("out/examples/chromatic_drift/chromatic_drift.png")
LANDSCAPE_OUTPUT = OUTPUT.with_stem("chromatic_drift_landscape")
DEFAULT_SEED = 19


def _linear(rgb):
    srgb = np.asarray(rgb, dtype=np.float32) / 255
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


PAPER = (*_linear((234, 226, 203)), 1.0)
PIGMENTS = _linear(
    ((15, 40, 54), (36, 74, 83), (61, 111, 114), (141, 174, 157), (185, 102, 43))
)


def _rows(size):
    # Each interleaved grid contributes half of the total rows.
    return 2 * ((size[1] - BOTTOM - TOP) // (2 * PITCH))


def _repeat_cells(values, size):
    """Expand cell data to canvas pixels in sdf-ui's bottom-left coordinates."""
    width, height = size
    pixels = np.zeros((height, width, values.shape[-1]), dtype=np.float32)
    pixels[BOTTOM : BOTTOM + values.shape[0] * PITCH, LEFT : LEFT + PRINT_WIDTH] = (
        values.repeat(PITCH, axis=0).repeat(PITCH, axis=1)
    )
    return pixels


def print_maps(seed, size=SIZE):
    """Make one texture for diamond geometry and another for pigment and wear."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, COLUMNS)[None, :]
    rows = _rows(size)
    y = np.linspace(0, 1, rows)[:, None]
    lattice = (rows, COLUMNS)
    drift = np.clip((x - 0.17) / 0.83, 0, 1)
    middle = np.sin(np.pi * drift)
    phase = 13 * y + 5 * x

    radius = 15.1 - 8.4 * drift + rng.normal(0, 1.55, lattice) * middle
    radius = np.clip(radius, 4, 16.3)
    dx = drift * (1.1 * np.sin(phase) + rng.uniform(-2.6, 2.6, lattice))
    dy = drift * (1.4 * np.cos(phase) + rng.uniform(-2.6, 2.6, lattice))
    skips = rng.random(lattice) < 0.13 * drift**2
    radius = np.where(skips, 0.05, radius)

    band = x + 0.13 * middle * rng.normal(size=lattice)
    band += 0.04 * np.sin(15 * y) * middle
    pigment = PIGMENTS[np.searchsorted((0.22, 0.37, 0.56, 0.70), band)]
    pigment *= (1 + middle * rng.uniform(-0.22, 0.24, lattice))[..., None]

    ink = _repeat_cells(
        np.concatenate((pigment, np.ones((*lattice, 1))), axis=-1), size
    )
    # Pinholes in the ink expose the paper. The red, green, and blue channels
    # stay constant in each stamp while opacity is sampled at each pixel.
    grain = rng.random(ink.shape[:2], dtype=np.float32)
    wear = rng.random(ink.shape[:2], dtype=np.float32)
    across = np.linspace(0, 1, size[0], dtype=np.float32)
    speckle = 0.026 + 0.075 * np.sin(np.pi * across) ** 2
    ink[..., 3] *= np.where(grain < speckle, wear**2, 0.98 + 0.02 * wear)

    # Neutral gray in alpha means no tilt. The right side gets a wider range
    # of angles, with a wave that carries through neighboring cells.
    sway = 0.68 * np.sin(phase) + 0.32 * rng.uniform(-1, 1, lattice)
    tilt = 0.5 + 0.5 * drift**1.25 * sway
    geometry = _repeat_cells(np.stack((dx, dy, radius, tilt), axis=-1), size)
    return geometry, ink


def chromatic_drift(geometry, pigment, size=SIZE):
    """Render overlapping diamonds using four interleaved repetition grids."""
    dx, dy, radius = (
        geometry.channel(channel).at_cell() for channel in ("red", "green", "blue")
    )
    rgb = tuple(
        pigment.channel(channel).at_cell() for channel in ("red", "green", "blue")
    )
    coverage = pigment.channel("alpha").at_pixel()
    tilt = geometry.channel("alpha").at_cell().remap((0, 1), (-pi / 4, pi / 4))
    half_side = radius / sqrt(2)
    stamp = sdf.rect((dx, dy), (half_side, half_side), angle=pi / 4 + tilt).ink_linear(
        (*rgb, coverage)
    )
    rows = _rows(size)
    print_size = (PRINT_WIDTH, rows * PITCH)
    cx, cy = LEFT + PRINT_WIDTH / 2, BOTTOM + print_size[1] / 2
    layers = (
        stamp.repeat_grid(
            COLUMNS // 2,
            rows // 2,
            (2 * PITCH, 2 * PITCH),
            (cx + x, cy + y),
        )
        for x in (-PITCH / 2, PITCH / 2)
        for y in (-PITCH / 2, PITCH / 2)
    )

    # Cover the outside of the print rectangle, giving the ink a clean edge.
    trim = sdf.rect_from_size((cx, cy), print_size).fill_linear(
        (*PAPER[:3], 0), PAPER, inner=-0.5, outer=0.5
    )
    title = sdf.text("CHROMATIC DRIFT", size=13, ox=LEFT, oy=50, letter_spacing=2.5)
    label = title.ink_linear((*PIGMENTS[0], 0.92))
    swatches = (
        sdf.diamond((704 - 14 * i, 54), (5, 5)).ink_linear((*ink, 1))
        for i, ink in enumerate(reversed(PIGMENTS))
    )
    return (
        color.linear_rgba(PAPER)
        .over(*layers, trim, label, *swatches)
        .finish(tone=0, dither=0.45)
    )


@contextmanager
def _texture(canvas, pixels):
    image = canvas.rgba32f()
    with ColorTexture(tex=image, context=canvas, mode="LINEAR") as texture:
        image.write(pixels)
        yield texture


def render(output=None, *, seed=DEFAULT_SEED, landscape=False):
    size = LANDSCAPE_SIZE if landscape else SIZE
    output = (
        Path(output)
        if output is not None
        else LANDSCAPE_OUTPUT
        if landscape
        else OUTPUT
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    geometry, pigment = print_maps(seed, size)
    with (
        Canvas(size) as canvas,
        _texture(canvas, geometry) as geometry_map,
        _texture(canvas, pigment) as pigment_map,
    ):
        canvas.save(chromatic_drift(geometry_map, pigment_map, size), output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Override the default output path")
    parser.add_argument(
        "--landscape", action="store_true", help="Render a 4:3 landscape print"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    print(render(args.output, seed=args.seed, landscape=args.landscape))


if __name__ == "__main__":
    main()
