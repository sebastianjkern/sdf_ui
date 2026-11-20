"""Geometry implementations for the chart grammar."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from ._plot_viewport import Plot
from .core.color import ColorTexture
from .core.expressions import _lazy_scalar

Rows = Sequence[Mapping[str, Any]]
X_CHANNELS = ("x", "xmin", "xmax")
Y_CHANNELS = ("y", "ymin", "ymax")


@dataclass(frozen=True)
class Geometry:
    """Describe a mark using normalized aesthetics and a canvas layer builder.

    Channels describe positions or bounds on each axis. The builder receives one group of
    mapped rows with categories encoded as positions, plus the group's tint.
    Return None when there is no mark to draw for that group.
    Set x_padding to the mark's half width in data units when numeric
    x limits should include the mark's edges.
    """

    channels: tuple[str, ...]
    draw: Callable[[Plot, Rows, Any], ColorTexture | None]
    include_zero: bool = False
    x_padding: float = 0.0

    def __post_init__(self):
        channels = set(self.channels)
        if not channels.intersection(X_CHANNELS) or not channels.intersection(
            Y_CHANNELS
        ):
            raise ValueError("geometry requires at least one channel on each axis")
        if channels.difference((*X_CHANNELS, *Y_CHANNELS)):
            raise ValueError(
                "geometry channels must be x, xmin, xmax, y, ymin, or ymax"
            )


def points(radius):
    def draw(plot, rows, tint):
        return plot.clip(
            plot.scatter(((row["x"], row["y"]) for row in rows), radius=radius)
        ).ink(tint)

    return Geometry(("x", "y"), draw)


def lines(width):
    def draw(plot, rows, tint):
        if len(rows) < 2:
            return None
        return plot.clip(
            plot.line(((row["x"], row["y"]) for row in rows), radius=width / 2)
        ).ink(tint)

    return Geometry(("x", "y"), draw)


def columns(width):
    def draw(plot, rows, tint):
        return plot.clip(
            plot.bars(((row["x"], row["y"]) for row in rows), width=width)
        ).ink(tint)

    return Geometry(
        ("x", "y"),
        draw,
        include_zero=True,
        x_padding=float(width) / 2 if isinstance(width, Real) else 0.0,
    )


def ranges(width, cap=None):
    half_cap = None if cap is None else _lazy_scalar(cap) / 2

    def draw(plot, rows, tint):
        def path():
            for row in rows:
                x, low, high = row["x"], row["ymin"], row["ymax"]
                yield from ((x, low), (x, high), None)
                if half_cap is not None:
                    for y in (low, high):
                        yield from ((x - half_cap, y), (x + half_cap, y), None)

        return plot.clip(plot.line(path(), radius=width / 2)).ink(tint)

    return Geometry(
        ("x", "ymin", "ymax"), draw, x_padding=cap / 2 if isinstance(cap, Real) else 0
    )


def rectangles(*, include_zero=False):
    def draw(plot, rows, tint):
        bounds = ((row["xmin"], row["xmax"], row["ymin"], row["ymax"]) for row in rows)
        return plot.clip(plot.rectangles(bounds)).ink(tint)

    return Geometry(("xmin", "xmax", "ymin", "ymax"), draw, include_zero=include_zero)


def ribbons():
    def draw(plot, rows, tint):
        if len(rows) < 2:
            return None
        samples = ((row["x"], row["ymin"], row["ymax"]) for row in rows)
        return plot.clip(plot.ribbon(samples)).ink(tint)

    return Geometry(("x", "ymin", "ymax"), draw)
