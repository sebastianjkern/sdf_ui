"""Render three plotting and math-graphics examples.

Run with ``uv run --extra math python -m examples.plot_gallery``.
"""

import argparse
import math
from functools import partial
from pathlib import Path

from sdf_ui import Canvas, Plot, color, sdf

OUTPUT_DIR = Path("out/examples/plot_gallery")
SIZE = (960, 640)
BACKGROUND = (0.035, 0.047, 0.075, 1.0)
WHITE = (0.94, 0.95, 0.97, 1.0)
MUTED = (0.53, 0.60, 0.70, 1.0)
GRID = (0.22, 0.28, 0.38, 0.62)
TRANSPARENT = (0.0, 0.0, 0.0, 0.0)


def _text(value, box, *, size=18, tint=WHITE, align="left"):
    return sdf.text(
        value,
        size=size,
        box=box,
        align=align,
        valign="middle",
        overflow="clip",
    ).ink(tint, inflate=0.2)


def _formula(value, center, *, size=24, tint=WHITE):
    return sdf.math(value, center=center, height=None, font_size=size, inflate=0.2).ink(
        tint
    )


def _heading(scene, title, subtitle):
    return scene.over(
        _text(title, (68, 565, 820, 34), size=20, tint=(0.45, 0.87, 0.83, 1.0))
    ).over(_text(subtitle, (68, 530, 820, 27), size=15, tint=MUTED))


def function_plot(canvas, output_dir):
    plot = Plot((132, 132, 710, 355), (-2 * math.pi, 2 * math.pi), (-7 / 5, 1.4))
    scene = _heading(
        color.clear(BACKGROUND),
        "01  FUNCTIONS AND ADAPTIVE CURVES",
        "Smoothly sampled functions, axes, ticks, and a data-anchored annotation",
    )
    scene = scene.over(
        plot.axes(
            x_ticks=(-157 / 25, -157 / 50, 0, 3.14, 6.28),
            y_ticks=(-1, 0, 1),
            axis_color=(0.74, 0.79, 0.87, 1.0),
            grid_color=GRID,
            label_color=MUTED,
            x_label="x",
            y_label="y",
        )
    )
    scene = scene.over(_function_curves(plot))
    scene = _function_annotations(scene, plot)
    canvas.save(scene, output_dir / "functions.png")


def _function_curves(plot):
    sine = plot.curve(math.sin, tolerance=0.18, samples=20).stroke(
        3.5, (0.28, 0.85, 0.77, 1.0)
    )
    cosine = plot.curve(math.cos, tolerance=0.18, samples=20).stroke(
        2.5, (0.98, 0.65, 0.34, 0.9)
    )
    return sine.over(cosine)


def _function_annotations(scene, plot):
    scene = scene.over(
        plot.annotate(
            (math.pi / 2, 1),
            "local maximum",
            offset=(18, 38),
            size=16,
            tint=(0.72, 0.94, 0.88, 1.0),
            leader_width=1.5,
        )
    )
    scene = scene.over(
        _formula(r"y=\sin x", (715, 508), size=22, tint=(0.28, 0.85, 0.77, 1.0))
    )
    scene = scene.over(
        _formula(r"y=\cos x", (825, 508), size=22, tint=(0.98, 0.65, 0.34, 1.0))
    )
    return scene


def parametric_gallery(canvas, output_dir):
    scene = _heading(
        color.clear(BACKGROUND),
        "02  PARAMETRIC CURVES",
        "Two parameterized paths, clipped to independent viewports",
    )
    plots = (
        Plot((105, 155, 330, 330), (-6 / 5, 6 / 5), (-6 / 5, 6 / 5)),
        Plot((525, 155, 330, 330), (-6 / 5, 6 / 5), (-6 / 5, 6 / 5)),
    )
    rose, lissajous = plots
    scene = scene.over(
        rose.axes(
            x_ticks=(-1, 0, 1),
            y_ticks=(-1, 0, 1),
            axis_color=MUTED,
            grid_color=GRID,
            label_color=MUTED,
        )
    )
    scene = scene.over(
        lissajous.axes(
            x_ticks=(-1, 0, 1),
            y_ticks=(-1, 0, 1),
            axis_color=MUTED,
            grid_color=GRID,
            label_color=MUTED,
        )
    )
    flower, loop = _parametric_curves(rose, lissajous)
    scene = scene.over(flower).over(loop)
    scene = _parametric_labels(scene)
    canvas.save(scene, output_dir / "parametric_curves.png")


def _parametric_curves(rose, lissajous):
    flower = rose.parametric(
        lambda t: (math.cos(5 * t) * math.cos(t), math.cos(5 * t) * math.sin(t)),
        (0, 2 * math.pi),
        samples=24,
        tolerance=0.2,
    ).stroke(3.2, (1.0, 0.49, 0.42, 1.0))
    loop = lissajous.parametric(
        lambda t: (math.sin(3 * t), math.sin(4 * t + math.pi / 2)),
        (0, 2 * math.pi),
        samples=48,
        tolerance=0.2,
    ).stroke(3.0, (0.38, 0.75, 1.0, 1.0))
    return flower, loop


def _parametric_labels(scene):
    scene = scene.over(
        _text(
            "FIVE PETALS",
            (105, 95, 330, 28),
            size=16,
            tint=(1.0, 0.61, 0.53, 1.0),
            align="center",
        )
    )
    scene = scene.over(
        _text(
            "LISSAJOUS  3:4",
            (525, 95, 330, 28),
            size=16,
            tint=(0.52, 0.81, 1.0, 1.0),
            align="center",
        )
    )
    scene = scene.over(_formula(r"r=\cos(5\theta)", (270, 76), size=21, tint=WHITE))
    scene = scene.over(
        _formula(r"(x,y)=(\sin 3t,\cos 4t)", (690, 76), size=21, tint=WHITE)
    )
    return scene


def optimization_plot(canvas, output_dir):
    plot = Plot((145, 91, 410, 410), (0, 1.2), (0, 1.2))
    scene = _heading(
        color.clear(BACKGROUND),
        "03  CONTOURS MEET A CONSTRAINT",
        "The maximum of xy on the unit circle occurs where the curves are tangent",
    )
    scene = scene.over(
        plot.axes(
            x_ticks=(0, 0.5, 1),
            y_ticks=(0.5, 1),
            axis_color=(0.74, 0.79, 0.87, 1.0),
            grid_color=GRID,
            label_color=MUTED,
            x_label="x",
            y_label="y",
        )
    )
    circle = plot.parametric(
        _unit_circle_point,
        (0, math.pi / 2),
        samples=24,
        tolerance=0.15,
    ).stroke(3.0, (0.26, 0.86, 0.75, 1.0))
    scene = scene.over(circle)
    scene = _optimization_contours(scene, plot)
    scene = _optimization_optimum(scene, plot)
    scene = _optimization_annotations(scene)
    canvas.save(scene, output_dir / "constraint_optimization.png")


def _optimization_contours(scene, plot):
    for level in (0.1, 0.2, 0.3, 0.4, 0.5):
        contour = plot.curve(
            partial(_hyperbola, level), domain=(level / (12 / 10), 12 / 10),
            tolerance=0.2, samples=20,
        ).stroke(1.5 if level < 0.5 else 3.0, (1.0, 0.68, 0.35, 0.82))
        scene = scene.over(contour)
    return scene


def _hyperbola(level, x):
    return level / x


def _unit_circle_point(angle):
    return math.cos(angle), math.sin(angle)


def _optimization_optimum(scene, plot):
    optimum = 1 / math.sqrt(2)
    scene = scene.over(
        plot.annotate(
            (optimum, optimum),
            "maximum: f = 1/2",
            offset=(25, 35),
            size=17,
            tint=(1.0, 0.83, 0.53, 1.0),
            leader_width=1.5,
        )
    )
    tangent = plot.parametric(
        lambda t: (optimum + t, optimum - t),
        (-11 / 50, 0.22),
        samples=4,
        tolerance=0.1,
    ).stroke(1.7, (0.72, 0.62, 1.0, 0.95))
    return scene.over(tangent)


def _optimization_annotations(scene):
    scene = scene.over(
        _formula(r"f(x,y)=xy", (765, 440), size=30, tint=(1.0, 0.75, 0.43, 1.0))
    )
    scene = scene.over(
        _formula(r"g(x,y)=x^2+y^2=1", (770, 385), size=25, tint=(0.36, 0.9, 0.8, 1.0))
    )
    scene = scene.over(
        _formula(
            r"\nabla f=\lambda\nabla g", (770, 320), size=28, tint=(0.78, 0.7, 1.0, 1.0)
        )
    )
    scene = scene.over(
        _text(
            "Level curves xy=c rise toward the highlighted contour.\nAt the optimum, the constraint and objective share a tangent.",
            (670, 205, 230, 76),
            size=17,
            tint=WHITE,
        )
    )
    return scene


def render_plot_gallery(output_dir=OUTPUT_DIR, only="all"):
    """Render function, parametric, and constrained-optimization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    renderers = {
        "functions": function_plot,
        "parametric": parametric_gallery,
        "optimization": optimization_plot,
    }
    selected = renderers if only == "all" else {only: renderers[only]}
    with Canvas(SIZE) as canvas:
        for render in selected.values():
            render(canvas, output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "functions", "parametric", "optimization"),
        default="all",
        help="Render one scene instead of the complete gallery",
    )
    args = parser.parse_args()
    print(render_plot_gallery(only=args.only))


if __name__ == "__main__":
    main()
