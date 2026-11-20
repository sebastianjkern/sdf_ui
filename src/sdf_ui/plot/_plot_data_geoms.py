"""Public data-geometry constructors for the layered chart grammar."""

from __future__ import annotations

import math
from numbers import Real

from ._plot_data import Layer
from ._plot_geometries import columns, lines, points, ranges, rectangles, ribbons
from ._plot_statistics import Histogram
from .core.expressions import _lazy_scalar


def geom_point(mapping=None, *, data=None, tint=None, radius=4.5) -> Layer:
    """Draw circles with a canvas-space radius."""
    return Layer(points(radius), mapping or {}, data, tint)


def geom_line(mapping=None, *, data=None, tint=None, width=2.5) -> Layer:
    """Connect rows in input order within each group using a canvas-space width."""
    return Layer(lines(_lazy_scalar(width)), mapping or {}, data, tint)


def geom_col(mapping=None, *, data=None, tint=None, width=0.8, dodge=0.0) -> Layer:
    """Draw columns from numeric y values, with width and dodge in x units.

    Dodge is the distance between adjacent groups' centers. Use a width
    smaller than this distance to put a gap between adjacent columns.
    """
    return Layer(columns(width), mapping or {}, data, tint, dodge)


def geom_bar(mapping=None, *, data=None, tint=None, width=0.8, dodge=0.0) -> Layer:
    """Count rows at each x within each group and draw the counts as columns."""
    return Layer(columns(width), mapping or {}, data, tint, dodge, count=True)


def geom_linerange(mapping=None, *, data=None, tint=None, width=2.0) -> Layer:
    """Draw vertical intervals from ymin to ymax using a canvas-space width."""
    return Layer(ranges(_lazy_scalar(width)), mapping or {}, data, tint)


def geom_errorbar(mapping=None, *, data=None, tint=None, width=2.0, cap=0.25) -> Layer:
    """Draw capped vertical intervals. Cap length is measured in x data units."""
    if isinstance(cap, Real) and (not math.isfinite(cap) or cap < 0):
        raise ValueError("error bar cap must be finite and nonnegative")
    return Layer(ranges(_lazy_scalar(width), cap), mapping or {}, data, tint)


def geom_rect(mapping=None, *, data=None, tint=None, alpha=None) -> Layer:
    """Draw rectangles from the xmin, xmax, ymin, and ymax aesthetics."""
    return Layer(rectangles(), mapping or {}, data, tint, alpha=alpha)


def geom_ribbon(mapping=None, *, data=None, tint=None, alpha=0.28) -> Layer:
    """Fill between ymin and ymax in row order within each group."""
    return Layer(ribbons(), mapping or {}, data, tint, alpha=alpha)


def geom_histogram(mapping=None, *, data=None, tint=None, bins=12, alpha=None) -> Layer:
    """Count finite numeric x observations in equal bins shared by the layer.

    Bins include the left edge and exclude the right edge, except that the
    last bin includes both. Groups in a panel overlay one another.
    """
    return Layer(
        rectangles(include_zero=True),
        mapping or {},
        data,
        tint,
        alpha=alpha,
        _histogram=Histogram(bins),
    )
