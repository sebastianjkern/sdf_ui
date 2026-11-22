from math import pi
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from sdf_ui import Canvas, HeightTexture, paint, sdf

SIZE = (240, 205)
CENTER = (120, 99)
PAPER = "#f5eee0"
INK = "#27465a"
LIGHT_X_MAGNITUDE = 0.85
LIGHT_X = -LIGHT_X_MAGNITUDE
LIGHT = (LIGHT_X, 0.5, 0.9)
OUTPUT = Path("out/examples/donut_hatch_lighting/comparison.png")
RADIUS = 76.0
RING_RADIUS = 42.0
RING_TUBE = 18.0


def _shape(is_ring):
    if is_ring:
        return sdf.ring(CENTER, RING_RADIUS, thickness=2 * RING_TUBE)
    return sdf.circle(CENTER, RADIUS)


def _height(canvas, is_ring):
    width, height = SIZE
    rows, columns = np.mgrid[:height, :width]
    rho = np.hypot(columns - CENTER[0], rows - CENTER[1])
    if is_ring:
        z = np.sqrt(np.maximum(RING_TUBE**2 - (rho - RING_RADIUS) ** 2, 0))
    else:
        z = np.sqrt(np.maximum(RADIUS**2 - rho**2, 0))
    texture = HeightTexture(tex=canvas.r32f(), context=canvas)
    texture.tex.write(z.astype(np.float32))
    return texture


def _style(depth=None, height_field=None):
    passes = (
        paint.HatchPass(pi / 4, 8, 1.0, min_shadow=0.18),
        paint.HatchPass(-pi / 4, 8, 1.0, min_shadow=0.42),
        paint.HatchPass(pi / 2, 7, 0.9, min_shadow=0.68),
    )
    light = (
        paint.HatchLight(direction=LIGHT, surface=height_field)
        if depth is None
        else paint.HatchLight(direction=LIGHT, depth=depth, surface=height_field)
    )
    return paint.HatchStyle(passes, light=light)


def _render_tile(canvas, shape, style, scratch):
    scene = paint.scene(
        paint.fill(shape, "#dfcfb5"),
        paint.fill(shape, INK, hatch=style),
        paint.outline(shape, INK, width=1.25),
        background=PAPER,
    )
    scene.save_png(scratch, canvas=canvas)
    with Image.open(scratch) as rendered:
        return rendered.copy()


def _comparison_tile(canvas, is_ring, mode, scratch):
    if mode == "default":
        return _render_tile(canvas, _shape(is_ring), _style(), scratch)
    depth = RING_TUBE if is_ring else RADIUS
    if mode == "matched":
        return _render_tile(canvas, _shape(is_ring), _style(depth), scratch)
    height_field = _height(canvas, is_ring)
    try:
        return _render_tile(
            canvas,
            _shape(is_ring),
            _style(depth, height_field),
            scratch,
        )
    finally:
        height_field.close()


def render(output=OUTPUT):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        ("Original · default depth 60", "default"),
        ("Original · depth = radius", "matched"),
        ("Analytic height · depth = radius", "analytic"),
    )
    rows = (("Disc", False), ("Ring", True))
    sheet = Image.new("RGB", (len(columns) * SIZE[0], len(rows) * SIZE[1] + 54), PAPER)
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (16, 10),
        "Disc radius 76 · ring radius 42 · tube radius 18 · thickness 36",
        fill=INK,
    )
    with Canvas(SIZE) as canvas:
        with TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "tile.png"
            for column, (label, mode) in enumerate(columns):
                draw.text((column * SIZE[0] + 12, 34), label, fill=INK)
                for row, (row_label, is_ring) in enumerate(rows):
                    tile = _comparison_tile(canvas, is_ring, mode, scratch)
                    left = column * SIZE[0]
                    top = 54 + row * SIZE[1]
                    sheet.paste(tile, (left, top))
                    draw.text((left + 12, top + 8), row_label, fill=INK)
    sheet.save(output)
    return output


if __name__ == "__main__":
    render()
