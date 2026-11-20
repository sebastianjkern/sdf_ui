"""Render standalone scatter, grouped bar, and diverging bar charts."""

from pathlib import Path

from _chart_helpers import chart_text

from sdf_ui import Canvas, Plot, color, sdf

OUTPUT = Path("out/examples/mark_gallery")
SIZE = (920, 540)
BOX = (100, 108, 720, 302)
TEAL = "#66e0c5"
GOLD = "#ffc880"
BLUE = "#8dbbff"
MUTED = "#9fb0c4"


def _page(title, subtitle):
    return color.clear("#101923").over(
        sdf.rect_from_size((460, 249), (832, 388), corner_radius=20).ink("#172637"),
        chart_text(title, 52, 482, 32),
        chart_text(subtitle, 54, 452, 16, MUTED),
    )


def _axes(plot, **options):
    return plot.axes(
        y_axis_at=plot.x_range[0],
        axis_color="#688098",
        grid_color="#314559",
        label_color=MUTED,
        **options,
    )


def _scatter():
    plot = Plot(BOX, (0, 8), (40, 100))
    sun = [
        (1, 48),
        (1.4, 55),
        (2.3, 52),
        (2.8, 65),
        (3.4, 61),
        (3.9, 74),
        (4.3, 68),
        (4.9, 79),
        (5.7, 76),
        (6.1, 87),
        (6.5, 82),
        (7.2, 94),
    ]
    shade = [
        (0.8, 53),
        (1.5, 60),
        (2.1, 56),
        (2.6, 62),
        (3.1, 58),
        (3.7, 70),
        (4.2, 66),
        (4.7, 61),
        (5.3, 72),
        (5.8, 64),
        (6.4, 75),
        (7.1, 69),
    ]
    return _page(
        "Light and growth", "Illustrative greenhouse readings · one point per plant"
    ).over(
        chart_text("Sun bench", 663, 452, 16, GOLD),
        chart_text("Shade bench", 755, 452, 16, TEAL),
        _axes(plot, x_ticks=range(0, 9, 2), y_ticks=range(40, 101, 15)),
        chart_text("Daily light (hours)", 396, 63, 16, MUTED),
        chart_text("Height (cm)", 100, 420, 14, MUTED),
        plot.scatter(shade, radius=5).ink(TEAL),
        plot.scatter(sun, radius=5).ink(GOLD),
        plot.annotate(
            sun[-1], "Tallest plant", offset=(-122, 7), tint=GOLD, leader_width=1.3
        ),
    )


def _grouped_bars():
    plot = Plot(BOX, (-11 / 20, 4.55), (0, 120))
    months = ("Feb", "Mar", "Apr", "May", "Jun")
    planned = (62, 66, 83, 90, 102)
    delivered = (55, 74, 76, 98, 109)
    return _page(
        "Orders shipped", "Monthly plan and actual shipments · illustrative data"
    ).over(
        chart_text("Plan", 700, 452, 16, BLUE),
        chart_text("Actual", 755, 452, 16, TEAL),
        _axes(
            plot,
            x_ticks=range(len(months)),
            y_ticks=range(0, 121, 30),
            x_format=months.__getitem__,
        ),
        plot.bars(
            ((i - 9 / 50, value) for i, value in enumerate(planned)), width=0.29
        ).ink(BLUE),
        plot.bars(
            ((i + 9 / 50, value) for i, value in enumerate(delivered)), width=0.29
        ).ink(TEAL),
    )


def _diverging_bars():
    plot = Plot(BOX, (-3 / 5, 5.6), (-3, 3))
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun")
    changes = (-17 / 10, 0.9, 1.6, -12 / 5, 2.6, 1.2)
    layers = []
    for positive, tint in ((True, TEAL), (False, GOLD)):
        points = [
            (i, value) for i, value in enumerate(changes) if (value > 0) == positive
        ]
        layers.append(plot.bars(points, width=0.5).ink(tint))
        layers.extend(
            plot.annotate(
                (i, value),
                f"{value:+g}",
                offset=(-15, 9 if positive else -20),
                size=15,
                tint=tint,
            )
            for i, value in points
        )
    return _page(
        "Net energy storage", "Monthly change in MWh · illustrative data"
    ).over(
        chart_text("Added", 681, 452, 16, TEAL),
        chart_text("Drawn", 755, 452, 16, GOLD),
        _axes(
            plot,
            x_ticks=range(len(months)),
            y_ticks=(-3, -3 / 2, 0, 1.5, 3),
            x_format=months.__getitem__,
        ),
        *layers,
    )


def render(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Canvas(SIZE) as canvas:
        for name, scene in (
            ("scatter", _scatter),
            ("grouped_bars", _grouped_bars),
            ("diverging_bars", _diverging_bars),
        ):
            canvas.save(scene(), output / f"{name}.png")
    return output


if __name__ == "__main__":
    print(render())
