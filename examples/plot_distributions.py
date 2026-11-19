"""Render histograms, uncertainty intervals, and shapes in data coordinates."""

from pathlib import Path
from random import Random

from sdf_ui import Canvas, color, sdf
from sdf_ui.plot import (
    Plot,
    aes,
    facet_wrap,
    geom_errorbar,
    geom_histogram,
    geom_line,
    geom_point,
    geom_rect,
    geom_ribbon,
    labs,
)

OUTPUT = Path("out/examples/plot_distributions")
SIZE = (960, 550)
BOX = (30, 25, 900, 500)


def _latency():
    rng = Random(17)
    rows = [
        {"profile": profile, "latency": rng.lognormvariate(center, spread)}
        for profile, center, spread in (
            ("Cached", 3.55, 0.19),
            ("Cold start", 4.11, 0.25),
        )
        for _ in range(240)
    ]
    return (
        Plot.from_data(rows, aes(x="latency", color="profile"))
        + geom_histogram(bins=16)
        + facet_wrap("profile")
        + labs(title="Request latency", x="Latency (ms) · simulated data", y="Requests")
    )


def _forecast():
    predictions = [
        {"hour": hour, "power": power, "low": power - error, "high": power + error}
        for hour, power, error in (
            (8, 1.1, 0.35),
            (9, 1.8, 0.4),
            (10, 2.9, 0.55),
            (11, 4.1, 0.6),
            (12, 5.0, 0.68),
            (13, 5.3, 0.72),
            (14, 4.7, 0.65),
            (15, 4.0, 0.6),
            (16, 2.8, 0.52),
            (17, 1.7, 0.43),
            (18, 0.9, 0.35),
        )
    ]
    sensors = [
        {"hour": hour, "power": power, "low": power - 0.17, "high": power + 0.17}
        for hour, power in ((9, 2.05), (12, 4.7), (15, 4.25), (17, 1.55))
    ]
    interval = aes(ymin="low", ymax="high")
    return (
        Plot.from_data(predictions, aes(x="hour", y="power"), ylim=(0, 6.5))
        + geom_ribbon(interval, tint="#8fb5ff", alpha=0.27)
        + geom_line(tint="#a4caff", width=2.6)
        + geom_errorbar(interval, data=sensors, tint="#ffc880", width=1.5, cap=0.21)
        + geom_point(data=sensors, tint="#ffc880", radius=4)
        + labs(
            title="Solar generation forecast",
            x="Hour of day · forecast band in blue, sensor readings in gold",
            y="Power (kW)",
        )
    )


def _schedule():
    windows = [
        {
            "start": start,
            "end": end,
            "low": lane - 0.24,
            "high": lane + 0.24,
            "kind": kind,
        }
        for lane, start, end, kind in (
            (1, 8.0, 10.8, "Production"),
            (1, 11.1, 12.3, "Maintenance"),
            (1, 12.6, 17.2, "Production"),
            (2, 8.8, 12.8, "Production"),
            (2, 14.1, 16.1, "Maintenance"),
            (3, 8.0, 9.5, "Maintenance"),
            (3, 10.1, 15.8, "Production"),
        )
    ]
    return (
        Plot.from_data(windows, aes(color="kind"), xlim=(7.5, 18), ylim=(0.5, 3.5))
        + geom_rect(aes(xmin="start", xmax="end", ymin="low", ymax="high"))
        + labs(title="Workshop schedule", x="Hour of day", y="Workstation")
    )


def _operating_envelope():
    plot = Plot((92, 105, 808, 338), (0, 12), (0, 100))
    # Separate runs show where the sensor was available. The rectangles
    # are periods flagged by a separate monitoring system.
    envelope = [
        (0, 22, 36),
        (1.4, 29, 43),
        (2.8, 33, 49),
        (4, 30, 45),
        None,
        (5.5, 37, 52),
        (7, 47, 63),
        (8.6, 51, 68),
        (10, 40, 55),
        (12, 34, 46),
    ]
    flags = [(3, 4.8, 0, 100), (8.3, 9.4, 0, 100)]
    return color.clear("#111c2b").over(
        sdf.text("Operating envelope", ox=46, oy=503, size=28).ink("#ecf1f9"),
        sdf.text(
            "Blue shows the available sensor range. Amber marks flagged periods.",
            ox=48,
            oy=473,
            size=15,
        ).ink("#a0b3c9"),
        plot.clip(plot.rectangles(flags)).ink("#ffc880").transparency(0.17),
        plot.axes(
            x_ticks=range(0, 13, 2),
            y_ticks=range(0, 101, 20),
            axis_color="#748ba1",
            grid_color="#34495d",
            label_color="#a0b3c9",
            label_size=13,
        ),
        plot.clip(plot.ribbon(envelope)).ink("#8fb5ff").transparency(0.65),
        sdf.text("Elapsed time (hours)", ox=48, oy=38, size=15).ink("#a0b3c9"),
    )


def render(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Canvas(SIZE) as canvas:
        for filename, plot in (
            ("latency", _latency()),
            ("solar_forecast", _forecast()),
            ("workshop_schedule", _schedule()),
        ):
            scene = color.clear("#111c2b").over(plot.scene(BOX))
            canvas.save(scene, output / f"{filename}.png")
        canvas.save(_operating_envelope(), output / "operating_envelope.png")
    return output


if __name__ == "__main__":
    print(render())
