"""Render a forecast chart using bars, a line series and data annotations."""

from pathlib import Path

from _chart_helpers import chart_text

from sdf_ui import Canvas, Plot, color, param, sdf

OUTPUT = Path("out/examples/series_plot.png")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun")
ACTUAL = (-7 / 5, 0.6, 1.5, 2.1, 1.8, 3.3)
FORECAST = (-4 / 5, 0.2, 1.1, 1.7, 2.3, param("june_forecast"))
TEAL = "#66e0c5"
GOLD = "#ffcb87"
MUTED = "#97aec0"


def _forecast_scene(plot, forecast):
    axes = plot.axes(
        x_ticks=range(len(MONTHS)), y_ticks=None, y_axis_at=plot.x_range[0],
        x_format=lambda index: MONTHS[index],
        y_format=lambda value: f"{value:g} MWh",
        axis_color="#6b859b", grid_color="#2b4053", label_color=MUTED,
    )
    annotation = plot.annotate(
        (5, FORECAST[-1]), "June forecast", offset=(-108, 30), tint=GOLD,
        leader_width=1.3,
    )
    data = (
        plot.clip(plot.bars(enumerate(ACTUAL), width=0.55)).ink("#288f83"),
        plot.clip(plot.line(forecast, radius=1.6)).ink(GOLD),
        plot.clip(plot.scatter(forecast, radius=4.5)).ink(GOLD),
    )
    labels = (
        chart_text("Energy storage forecast", 64, 518, 31, "#f5f4ef"),
        chart_text("Monthly change in stored energy", 66, 484, 17, MUTED),
        chart_text("Measured", 647, 487, 16, TEAL),
        chart_text("Forecast", 755, 487, 16, GOLD),
    )
    panel = sdf.rect_from_size((480, 263), (868, 416), corner_radius=20).ink("#162333")
    return color.clear("#101824").over(panel, *labels, axes, *data, annotation)


def render(output=OUTPUT):
    plot = Plot((110, 99, 728, 345), (-3 / 5, 5.6), (-2, 4))
    scene = _forecast_scene(plot, tuple(enumerate(FORECAST)))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Canvas((960, 570)) as canvas:
        canvas.save(scene, output, params={"june_forecast": 2.8})
    return output


if __name__ == "__main__":
    print(render())
