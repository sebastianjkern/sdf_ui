"""Render a forecast chart using bars, a line series and data annotations."""

from pathlib import Path

from sdf_ui import Canvas, Plot, color, param, sdf

OUTPUT = Path("out/examples/series_plot.png")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun")
ACTUAL = (-1.4, 0.6, 1.5, 2.1, 1.8, 3.3)
FORECAST = (-0.8, 0.2, 1.1, 1.7, 2.3, param("june_forecast"))
TEAL = "#66e0c5"
GOLD = "#ffcb87"
MUTED = "#97aec0"


def _caption(text, x, y, size, tint):
    return sdf.text(text, ox=x, oy=y, size=size).ink(tint, inflate=0.2)


def render(output=OUTPUT):
    plot = Plot((110, 99, 728, 345), (-0.6, 5.6), (-2, 4))
    forecast = tuple(enumerate(FORECAST))
    scene = color.clear("#101824").over(
        sdf.rect_from_size((480, 263), (868, 416), corner_radius=20).ink("#162333"),
        _caption("Energy storage forecast", 64, 518, 31, "#f5f4ef"),
        _caption("Monthly change in stored energy", 66, 484, 17, MUTED),
        _caption("Measured", 647, 487, 16, TEAL),
        _caption("Forecast", 755, 487, 16, GOLD),
        plot.axes(
            x_ticks=range(len(MONTHS)),
            y_ticks=None,
            y_axis_at=plot.x_range[0],
            x_format=lambda index: MONTHS[index],
            y_format=lambda value: f"{value:g} MWh",
            axis_color="#6b859b",
            grid_color="#2b4053",
            label_color=MUTED,
        ),
        plot.clip(plot.bars(enumerate(ACTUAL), width=0.55)).ink("#288f83"),
        plot.clip(plot.line(forecast, radius=1.6)).ink(GOLD),
        plot.clip(plot.scatter(forecast, radius=4.5)).ink(GOLD),
        plot.annotate(
            (5, FORECAST[-1]),
            "June forecast",
            offset=(-108, 30),
            tint=GOLD,
            leader_width=1.3,
        ),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Canvas((960, 570)) as canvas:
        canvas.save(scene, output, params={"june_forecast": 2.8})
    return output


if __name__ == "__main__":
    print(render())
