"""Render a chart with layered intervals and another with grouped counts."""

from pathlib import Path

from sdf_ui import Canvas, color
from sdf_ui.plot import (
    Plot,
    aes,
    facet_wrap,
    geom_bar,
    geom_line,
    geom_linerange,
    geom_point,
    labs,
)

OUTPUT = Path("out/examples/chart_grammar")
SIZE = (960, 560)
BOX = (32, 26, 896, 508)


def _temperature_chart():
    series = (
        ("Greenhouse east", "Open vents", (19, 21, 23, 25, 23, 20)),
        ("Greenhouse east", "Closed vents", (19, 23, 27, 31, 29, 24)),
        ("Greenhouse west", "Open vents", (18, 20, 23, 24, 22, 19)),
        ("Greenhouse west", "Closed vents", (18, 22, 26, 29, 27, 22)),
    )
    readings = [
        {
            "house": house,
            "vents": vents,
            "time": hour,
            "temperature": value,
            "low": value - spread,
            "high": value + spread,
        }
        for house, vents, values in series
        for hour, value, spread in zip(
            range(8, 19, 2), values, (0.5, 1, 1.5, 1, 1.5, 0.5)
        )
    ]
    return (
        Plot.from_data(
            readings, aes(x="time", y="temperature", color="vents"), ylim=(16, 34)
        )
        + geom_linerange(aes(ymin="low", ymax="high"), width=1.25)
        + geom_line(width=2)
        + geom_point(radius=4)
        + facet_wrap("house")
        + labs(
            title="Ventilation and greenhouse temperature",
            x="Hour of day",
            y="Temperature (°C)",
        )
    )


def _shipments_chart():
    counts = (
        ("Mon", 11, 6),
        ("Tue", 15, 8),
        ("Wed", 17, 11),
        ("Thu", 13, 16),
        ("Fri", 18, 12),
    )
    parcels = [
        {"day": day, "service": service}
        for day, standard, express in counts
        for service, n in (("Standard", standard), ("Express", express))
        for _ in range(n)
    ]
    return (
        Plot.from_data(parcels, aes(x="day", color="service"))
        + geom_bar(width=0.28, dodge=0.34)
        + labs(title="Parcel volume by service", x="Shipping day", y="Parcels")
    )


def render(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Canvas(SIZE) as canvas:
        for name, chart in (
            ("ventilation", _temperature_chart()),
            ("parcel_volume", _shipments_chart()),
        ):
            scene = color.clear("#111c2b").over(chart.scene(BOX))
            canvas.save(scene, output / f"{name}.png")
    return output


if __name__ == "__main__":
    print(render())
