from math import pi
from pathlib import Path

from sdf_ui import Canvas, paint, param, sdf
from sdf_ui.export import PlotterProfile

OUTPUT = Path("out/examples/crosshatch/hand")
SIZE = (300, 284)
PAPER = "#f8f1e5"
INK = "#273b49"
LIGHT_OFFSET = 0.8


def _scene(shape, style):
    return paint.scene(
        paint.fill(shape, INK, hatch=style),
        paint.outline(shape, INK, width=1.1),
        background=PAPER,
    )


def _styles():
    first = paint.HatchPass(pi / 5, 8, 1.15)
    second = paint.HatchPass(-pi / 3, 11, 1.1)
    hand = paint.HandStroke(bow=4.2, jitter=1.15, seed=11)
    light = paint.HatchLight(
        direction=(param("light_x", -LIGHT_OFFSET), 0.9, 0.75), depth=102
    )
    shading = (
        paint.HatchPass(first.angle, 7.5, 1.15, min_shadow=0.27),
        paint.HatchPass(second.angle, 8.0, 1.1, min_shadow=0.58),
        paint.HatchPass(pi / 2, 8.5, 1.05, phase=2.0, min_shadow=0.83),
    )
    return (
        paint.HatchStyle((first, second)),
        paint.HatchStyle((first, second), hand=hand),
        paint.HatchStyle(shading, hand=hand, light=light),
    )


def hand_hatch_example(output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    sphere = sdf.circle((150, 142), 102)
    plotter = PlotterProfile((150, 142), 0.5, "G0 Z3", "G0 Z0", 1600, 700)
    flat, hand, lit = (_scene(sphere, style) for style in _styles())
    with Canvas(SIZE) as canvas:
        flat.save_png(output / "flat.png", canvas=canvas)
        hand.save_png(output / "hand.png", canvas=canvas)
        hand.save_svg(output / "hand.svg", canvas=canvas)
        lit.save_png(output / "lit.png", canvas=canvas)
        lit.save_svg(output / "lit.svg", canvas=canvas)
        lit.save_gcode(output / "lit.gcode", canvas=canvas, plotter=plotter)
        lit.save_png(
            output / "lit_reverse.png", canvas=canvas, params={"light_x": LIGHT_OFFSET}
        )


if __name__ == "__main__":
    hand_hatch_example()
