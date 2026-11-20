"""Letter the SDF UI banner with parametric curves on a plot."""

import argparse
from itertools import pairwise
from pathlib import Path

from sdf_ui import Canvas, Plot, color, sdf

SIZE = (1600, 640)
OUTPUT = Path("out/examples/parametric_banner/banner.png")

# Segments have either two endpoints or four Bézier control points.
# Each letter uses local coordinates with the baseline near y = 0.
S = (
    (
        ((1.02, 1.24), (0.80, 1.51), (0.10, 1.49), (0.10, 1.06)),
        ((0.10, 1.06), (0.10, 0.83), (0.34, 0.78), (0.61, 0.72)),
        ((0.61, 0.72), (0.90, 0.65), (1.06, 0.54), (1.06, 0.34)),
        ((1.06, 0.34), (1.06, -0.01), (0.42, -0.10), (0.06, 0.17)),
    ),
)
D = (
    (
        ((0.06, 0.05), (0.06, 1.40)),
        ((0.06, 1.40), (0.50, 1.40)),
        ((0.50, 1.40), (1.47, 1.40), (1.47, 0.05), (0.50, 0.05)),
        ((0.50, 0.05), (0.06, 0.05)),
    ),
)
F = (
    (((0.08, 0.05), (0.08, 1.40)), ((0.08, 1.40), (1.00, 1.40))),
    (((0.08, 0.79), (0.87, 0.79)),),
)
U = (
    (
        ((0.08, 1.40), (0.08, 0.56)),
        ((0.08, 0.56), (0.08, -0.13), (1.18, -0.13), (1.18, 0.56)),
        ((1.18, 0.56), (1.18, 1.40)),
    ),
)
GLYPH_I = (
    (((0.05, 1.40), (0.79, 1.40)),),
    (((0.42, 1.40), (0.42, 0.05)),),
    (((0.05, 0.05), (0.79, 0.05)),),
)


def _bezier(points, t):
    """Evaluate a line or a cubic by interpolating its control polygon."""
    while len(points) > 1:
        points = tuple(
            (x + (u - x) * t, y + (v - y) * t) for ((x, y), (u, v)) in pairwise(points)
        )
    return points[0]


PAPER = "#f0ebe1"
PLOT_SURFACE = "#fbf8f1"
AXIS = "#4b5964"
GRID = "#dadbd5"
LINE = "#223f58"
CUBIC = "#b84b34"
GLYPHS = ((0.32, S), (1.72, D), (3.26, F), (4.95, U), (6.57, GLYPH_I))
BASELINE = 0.38


def _trace(plot, points):
    def curve(t):
        return _bezier(points, t)

    return plot.parametric(curve, (0, 1), samples=8, tolerance=0.14, radius=4.9)


def _letter_layers(plot):
    paths = {2: [], 4: []}
    knots = set()
    for offset, glyph in GLYPHS:
        for stroke in glyph:
            for segment in stroke:
                points = tuple((x + offset, y + BASELINE) for x, y in segment)
                paths[len(points)].append(_trace(plot, points))
                knots.update((points[0], points[-1]))

    knot_positions = sorted(knots)
    return (
        sdf.union_all(paths[2]).ink(LINE),
        sdf.union_all(paths[4]).ink(CUBIC),
        plot.scatter(knot_positions, radius=4.8).ink(PLOT_SURFACE),
        plot.scatter(knot_positions, radius=2.2).ink(LINE),
    )


def _label(value, x, y, size, tint=AXIS):
    return sdf.text(value, size=size, ox=x, oy=y).ink(tint)


def _legend():
    y = 563
    line_key = sdf.polyline(((934, y), (970, y)), radius=3.3).ink(LINE)
    cubic_key = sdf.polyline(((1136, y), (1172, y)), radius=3.3).ink(CUBIC)
    knot_key = sdf.circle((1355, y), 5).stroke(2.0, LINE)
    return (
        line_key,
        _label("Line segment", 982, y - 7, 22),
        cubic_key,
        _label("Cubic Bézier", 1184, y - 7, 22),
        knot_key,
        _label("Knot", 1374, y - 7, 22),
    )


def banner():
    """Draw the wordmark as two data series on a labeled Cartesian plot."""
    plot = Plot((142, 116, 1320, 390), (0, 7.7), (0, 2.2))
    axes = plot.axes(
        x_ticks=range(8),
        y_ticks=(0, 0.5, 1, 1.5, 2),
        axis_color=AXIS,
        grid_color=GRID,
        label_color=AXIS,
        axis_width=1.6,
        grid_width=0.8,
        label_size=24,
        x_label="x(t)",
        y_label="y(t)",
    )
    surface = sdf.rect_from_size((802, 311), (1320, 390))
    frame = surface.stroke(1.0, GRID)
    return color.clear(PAPER).over(
        surface.ink(PLOT_SURFACE),
        frame,
        axes,
        *_letter_layers(plot),
        _label("Parametric letterforms", 142, 550, 38, LINE),
        *_legend(),
    )


def render(output=OUTPUT):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Canvas(SIZE) as canvas:
        canvas.save(banner(), output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    print(render(parser.parse_args().output))


if __name__ == "__main__":
    main()
