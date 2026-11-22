import math
from pathlib import Path

from sdf_ui import Canvas, paint, param, sdf
from sdf_ui.export import PlotterProfile

OUTPUT_DIR = Path("out/examples/crosshatch")
DARK = "#233a49"
WARM = "#af562d"
PAPER = "#fbf7ed"
CROSSHATCH = paint.HatchStyle(
    (paint.HatchPass(math.pi / 4, 9, 1.25), paint.HatchPass(-math.pi / 4, 14, 1))
)


def _crescent_layers(center):
    x, y = center
    moon = sdf.circle(center, param("moon_radius", 71)).subtract(
        sdf.circle((x + 22, y + 17), 47)
    )
    return (
        paint.fill(moon, WARM, hatch=CROSSHATCH, pen="copper"),
        paint.outline(moon, WARM, width=1.6, pen="copper"),
    )


def _mountain_layers():
    mountain = sdf.triangle((45, 106), (216, 367), (386, 106)).smooth_union(
        sdf.triangle((240, 106), (378, 287), (567, 106)), k=14
    )
    snow = mountain.intersection(sdf.triangle((122, 250), (213, 384), (312, 250)))
    rock = mountain.subtract(snow)
    water = sdf.rect_from_size((307, 72), (522, 26), 12)
    light = paint.HatchStyle((paint.HatchPass(math.pi / 4, 10, 1),))
    horizontal = paint.HatchStyle((paint.HatchPass(0, 6, 1, 2),))
    return (
        paint.fill(rock, DARK, hatch=CROSSHATCH, pen="blue"),
        paint.fill(snow, DARK, hatch=light, pen="blue"),
        paint.outline(mountain, DARK, width=1.6, pen="blue"),
        paint.fill(water, DARK, hatch=horizontal, pen="blue"),
    )


def crosshatch_example(output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    drawing = paint.scene(
        *_crescent_layers((475, 335)), *_mountain_layers(), background=PAPER
    )
    plotter = PlotterProfile(
        size_mm=(160, 120),
        mm_per_unit=0.25,
        pen_up_command="G0 Z5",
        pen_down_command="G0 Z0",
        travel_speed_mm_min=2400,
        draw_speed_mm_min=950,
        pen_change_commands={"copper": "M0", "blue": "M0"},
    )
    with Canvas((640, 480)) as canvas:
        drawing.save_png(output_dir / "mountain.png", canvas=canvas)
        drawing.save_svg(output_dir / "mountain.svg", canvas=canvas)
        drawing.save_gcode(
            output_dir / "mountain.gcode", canvas=canvas, plotter=plotter
        )
    with Canvas((240, 240)) as canvas:
        paint.scene(*_crescent_layers((120, 120)), background=PAPER).save_png(
            output_dir / "crescent.png", canvas=canvas, params={"moon_radius": 85}
        )


if __name__ == "__main__":
    crosshatch_example()
