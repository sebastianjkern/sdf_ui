"""Turn a procedural design or a photo into an adaptive rounded-square mosaic."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from sdf_ui import Canvas, FieldTexture, color, param, sdf
from sdf_ui.core.expressions import _lazy_scalar, _select_positive

SIZE = (576, 768)
TILE_SIZES = (64, 32, 16, 8)
MAX_DEPTH = len(TILE_SIZES) - 1
OUTPUT = Path("out/generative_art/quadtree_mosaic.png")


def orbital_guide():
    """Draw an orbital pattern with smooth silhouettes and finer bands."""
    dark = (0.012, 0.012, 0.012, 1)
    light = (0.92, 0.92, 0.92, 1)
    center = (293, 410)
    bands = sdf.union_all(
        sdf.ring(center, radius, thickness=thickness)
        for radius, thickness in ((212, 19), (241, 12), (267, 7), (285, 4))
    )
    planet = sdf.circle(center, 167)
    crescent = planet.subtract(sdf.circle((344, 467), 152))
    meridians = (
        sdf.union_all(
            sdf.ring((344, 467), radius, thickness=7) for radius in (75, 96, 117, 138)
        )
        .intersection(planet)
        .subtract(crescent)
    )
    satellite = sdf.circle((101, 613), 54)
    moon = sdf.circle((457, 143), 79)
    moon_glint = moon.subtract(sdf.circle((477, 162), 68))

    return color.linear_rgba(light).over(
        bands.ink_linear(dark),
        planet.ink_linear(dark),
        crescent.ink_linear(light),
        meridians.ink_linear(light),
        satellite.ink_linear(dark),
        sdf.circle((116, 629), 26).ink_linear(light),
        moon.ink_linear(dark),
        moon_glint.ink_linear(light),
    )


def read_guide(canvas, image_path):
    if image_path is None:
        with canvas.render(orbital_guide()) as guide:
            return guide.read()[..., 0]
    with Image.open(image_path) as source:
        image = ImageOps.fit(
            ImageOps.exif_transpose(source).convert("L"), SIZE, Image.Resampling.LANCZOS
        )
        # The engine's canvas y coordinate increases upward.
        return np.asarray(image, dtype=np.float32)[::-1] / 255


def partition(luminance, contrast):
    """Build a parameter map with one channel per subdivision level."""
    height, width = luminance.shape
    controls = np.zeros((height, width, len(TILE_SIZES)), dtype=np.float32)

    def visit(x, y, level):
        edge = TILE_SIZES[level]
        region = luminance[y : y + edge, x : x + edge]
        split = level < MAX_DEPTH and region.std() > contrast
        # Magnitude encodes tone. A negative value marks a node with children.
        tone = 0.25 if region.mean() < 0.5 else 1.0
        controls[y : y + edge, x : x + edge, level] = -tone if split else tone
        if split:
            half = edge // 2
            for dy in (0, half):
                for dx in (0, half):
                    visit(x + dx, y + dy, level + 1)

    for y in range(0, height, TILE_SIZES[0]):
        for x in range(0, width, TILE_SIZES[0]):
            visit(x, y, 0)
    return controls


def mosaic(controls, *, max_depth=param("max_depth", MAX_DEPTH)):
    """Render adaptive leaves up to a render-time depth limit."""
    max_depth = _lazy_scalar(max_depth)
    width, height = SIZE
    layers = []
    for level, edge in enumerate(TILE_SIZES):
        cell = controls.channel(level).at_cell()
        # At the depth cap, draw the parent using its own average tone.
        visible_level = (
            1 if level == 0 else _select_positive(max_depth - level + 0.5, 1, 0)
        )
        show_parent = _select_positive(max_depth - level - 0.5, 0, 1)
        opacity = visible_level * (
            (4 * cell).clamp() + show_parent * (-4 * cell).clamp()
        )
        tone = abs(cell).remap((0.25, 1.0), (0.006, 0.94))
        inset = 0.65 if edge >= 16 else 0.5
        tile = sdf.rounded_rect_from_size(
            (0, 0), (edge - 2 * inset, edge - 2 * inset), corner_radius=edge * 0.16
        ).ink_linear((tone, tone, tone, opacity))
        layers.append(
            tile.repeat_grid(
                width // edge, height // edge, edge, (width / 2, height / 2)
            )
        )
    return (
        color.linear_rgba((0.18, 0.17, 0.155, 1)).over(*layers).finish(tone=0, dither=0)
    )


def render(output=OUTPUT, *, image_path=None, contrast=0.16, max_depth=MAX_DEPTH):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Canvas(SIZE) as canvas:
        data = partition(read_guide(canvas, image_path), contrast)
        with FieldTexture(tex=canvas.rgba32f(), context=canvas) as controls:
            controls.tex.write(data)
            canvas.save(mosaic(controls), output, params={"max_depth": max_depth})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, help="Use an image as the luminance guide."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--max-depth",
        type=int,
        choices=range(MAX_DEPTH + 1),
        default=MAX_DEPTH,
        help="Maximum subdivision depth. Zero keeps the 64-pixel grid.",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=0.16,
        help="Local contrast needed to split a tile.",
    )
    args = parser.parse_args()
    if not 0 <= args.contrast <= 0.5:
        parser.error("--contrast must be between 0 and 0.5")
    print(
        render(
            args.output,
            image_path=args.image,
            contrast=args.contrast,
            max_depth=args.max_depth,
        )
    )


if __name__ == "__main__":
    main()
