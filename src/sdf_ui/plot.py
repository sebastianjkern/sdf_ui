"""Cartesian plots built from composable SDF UI layers."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from numbers import Integral, Real
from typing import Any

from .core.color import ColorTexture
from .core.expressions import Expr, _lazy_scalar
from .core.sdf import SDFTexture
from .util import hex_col

Point = tuple[float, float]
Box = tuple[float, float, float, float]
_MAX_DEPTH = 10


def _ordered_range(values: Point, name: str) -> Point:
    low, high = map(float, values)
    if not math.isfinite(high - low) or not low < high:
        raise ValueError(f"{name} must contain two increasing finite values")
    return low, high


def _clip_segment(a: Point, b: Point, low: Point, high: Point):
    """Clip a segment to a rectangle using its parameter interval."""
    first, last = 0.0, 1.0
    for start, end, lower, upper in zip(a, b, low, high):
        delta = end - start
        if delta == 0:
            if start < lower or start > upper:
                return None
            continue
        enter, exit_ = sorted(((lower - start) / delta, (upper - start) / delta))
        first, last = max(first, enter), min(last, exit_)
        if first > last:
            return None

    def point_at(t):
        if t == 0.0:
            return a
        if t == 1.0:
            return b
        return tuple(
            min(upper, max(lower, start + t * (end - start)))
            for start, end, lower, upper in zip(a, b, low, high)
        )

    return point_at(first), point_at(last)


def _adaptive_segments(sample, start, a, end, b, tolerance, depth=0):
    mid = start + (end - start) / 2
    times = (start + (mid - start) / 2, mid, mid + (end - mid) / 2)
    probes = tuple(sample(t) for t in times)
    points = (a, *probes, b)
    if all(point is None for point in points):
        return
    if all(point is not None for point in points):
        # Measure interpolation error in canvas units. A perpendicular-only
        # test can accept a near-vertical chord through a finite jump.
        if all(
            math.hypot(
                p[0] - (a[0] + (b[0] - a[0]) * ratio),
                p[1] - (a[1] + (b[1] - a[1]) * ratio),
            )
            <= tolerance
            for p, ratio in zip(probes, (0.25, 0.5, 0.75))
        ):
            yield a, b
            return
    # An unresolved interval may contain a discontinuity. Leave a gap.
    if depth < _MAX_DEPTH and start < mid < end:
        yield from _adaptive_segments(
            sample, start, a, mid, probes[1], tolerance, depth + 1
        )
        yield from _adaptive_segments(
            sample, mid, probes[1], end, b, tolerance, depth + 1
        )


def _path_points(segments):
    points = []
    previous = None
    for a, b in segments:
        if previous != a:
            if previous is not None:
                points.append(None)
            points.append(a)
        points.append(b)
        previous = b
    return tuple(points)


def _transparent_ink(tint):
    if isinstance(tint, str):
        tint = hex_col(tint)
    if isinstance(tint, (tuple, list)):
        return (*tint[:3], 0.0)
    return (*(Expr("index", (tint, index)) for index in range(3)), 0.0)


@dataclass(frozen=True)
class Plot:
    """Map graph coordinates into a lower-left-origin canvas viewport.

    ``box`` is ``(x, y, width, height)`` in canvas units. The x and y ranges
    are independent, so a unit can have a different visual scale on each axis.
    Compose the output of ``axes()`` and any number of curves as normal layers.
    """

    box: Box
    x_range: Point
    y_range: Point

    def __post_init__(self):
        x, y, width, height = map(float, self.box)
        if not all(map(math.isfinite, (x, y, x + width, y + height))):
            raise ValueError("plot box must contain finite canvas coordinates")
        if width <= 0 or height <= 0:
            raise ValueError("plot box width and height must be positive")
        object.__setattr__(self, "box", (x, y, width, height))
        object.__setattr__(self, "x_range", _ordered_range(self.x_range, "x_range"))
        object.__setattr__(self, "y_range", _ordered_range(self.y_range, "y_range"))

    def point(self, x: Any, y: Any) -> tuple[Any, Any]:
        """Map a graph coordinate or lazy data expression to the canvas."""
        left, bottom, width, height = self.box
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        x, y = _lazy_scalar(x), _lazy_scalar(y)
        return (
            left + (x - x_min) / (x_max - x_min) * width,
            bottom + (y - y_min) / (y_max - y_min) * height,
        )

    def curve(
        self,
        function: Callable[[float], float | None],
        *,
        domain: Point | None = None,
        tolerance: float = 0.25,
        samples: int = 32,
        breaks: Iterable[float] = (),
        radius: float = 0.0,
    ) -> SDFTexture:
        """Plot y=f(x) over the visible x range or a smaller domain.

        Return unsigned distance to the sampled line. Set a positive radius
        for a filled distance band, or use ``.stroke()`` on the centerline.
        """
        from . import sdf

        def point_at(x):
            y = function(x)
            return None if y is None else (x, y)

        start, end = self.x_range
        if domain is not None:
            low, high = _ordered_range(domain, "domain")
            start, end = max(low, start), min(high, end)
        if start >= end:
            return sdf.polyline((), radius)
        return self.parametric(
            point_at,
            (start, end),
            tolerance=tolerance,
            samples=samples,
            breaks=breaks,
            radius=radius,
        )

    def parametric(
        self,
        function: Callable[[float], Point | None],
        domain: Point,
        *,
        tolerance: float = 0.25,
        samples: int = 32,
        breaks: Iterable[float] = (),
        radius: float = 0.0,
    ) -> SDFTexture:
        """Plot (x(t), y(t)) and clip its centerline to the viewport.

        Start with ``samples`` intervals per continuous part and refine where
        interpolation error exceeds ``tolerance`` canvas units. Increase
        ``samples`` for very oscillatory functions. Non-finite values, domain
        errors and ``None`` leave gaps. Pass known jumps or poles in ``breaks``
        so sampling never connects across them.
        """
        from . import sdf

        start, end = _ordered_range(domain, "domain")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("plot tolerance must be finite and positive")
        if (
            isinstance(samples, bool)
            or not isinstance(samples, Integral)
            or samples < 1
        ):
            raise ValueError("plot samples must be a positive integer")
        breaks = tuple(map(float, breaks))
        if not all(map(math.isfinite, breaks)):
            raise ValueError("plot breaks must be finite")
        cuts = (start, *sorted({t for t in breaks if start < t < end}), end)
        left, bottom, width, height = self.box
        low, high = (left, bottom), (left + width, bottom + height)

        @cache
        def sample(t):
            try:
                value = function(t)
            except (ArithmeticError, ValueError):
                return None
            if value is None:
                return None
            p = self.point(*map(float, value))
            return p if all(map(math.isfinite, p)) else None

        def clipped_segments():
            for index, (a, b) in enumerate(zip(cuts, cuts[1:])):
                # Take one-sided samples at explicitly marked discontinuities.
                lo = a if index == 0 else math.nextafter(a, b)
                hi = b if index == len(cuts) - 2 else math.nextafter(b, a)
                if lo >= hi:
                    continue
                previous_t = lo
                previous_p = sample(lo)
                for i in range(1, samples + 1):
                    t = hi if i == samples else lo + (hi - lo) * (i / samples)
                    point = sample(t)
                    for p0, p1 in _adaptive_segments(
                        sample, previous_t, previous_p, t, point, tolerance
                    ):
                        clipped = _clip_segment(p0, p1, low, high)
                        if clipped is not None:
                            yield clipped
                    previous_t, previous_p = t, point

        return sdf.polyline(_path_points(clipped_segments()), radius)

    def axes(
        self,
        x_ticks: Iterable[float] = (),
        y_ticks: Iterable[float] = (),
        *,
        axis_color="#a4adbc",
        grid_color=(0.35, 0.40, 0.49, 0.45),
        label_color=None,
        axis_width: float = 2.0,
        grid_width: float = 1.0,
        label_size: float = 15.0,
        x_label: str = "",
        y_label: str = "",
    ) -> ColorTexture:
        """Draw the axes, optional grid lines and labels as a transparent layer."""
        from . import color, sdf

        xmin, xmax = self.x_range
        ymin, ymax = self.y_range
        left, bottom, width, height = self.box
        right, top = left + width, bottom + height
        x_axis = 0.0 if ymin <= 0 <= ymax else ymin
        y_axis = 0.0 if xmin <= 0 <= xmax else xmin
        axis_x, axis_y = self.point(y_axis, x_axis)
        xt = [
            (value, self.point(value, ymin)[0])
            for value in x_ticks
            if xmin <= value <= xmax
        ]
        yt = [
            (value, self.point(xmin, value)[1])
            for value in y_ticks
            if ymin <= value <= ymax
        ]

        grid = _path_points(
            [((x, bottom), (x, top)) for value, x in xt if value != y_axis]
            + [((left, y), (right, y)) for value, y in yt if value != x_axis]
        )
        layer = color.clear((0.0, 0.0, 0.0, 0.0))
        if grid:
            layer = layer.alpha_overlay(
                sdf.polyline(grid).stroke(grid_width, grid_color)
            )
        lines = ((left, axis_y), (right, axis_y), None, (axis_x, bottom), (axis_x, top))
        layer = layer.alpha_overlay(sdf.polyline(lines).stroke(axis_width, axis_color))

        tint = axis_color if label_color is None else label_color
        transparent = _transparent_ink(tint)

        def label(value, x, y, align):
            label_width = max(1, len(value)) * label_size
            if align == "center":
                x -= label_width / 2
            elif align == "right":
                x -= label_width
            return sdf.text(
                value,
                size=label_size,
                box=(x, y, label_width, 1.5 * label_size),
                align=align,
            ).fill(tint, transparent)

        for value, x in xt:
            layer = layer.alpha_overlay(
                label(f"{value:g}", x, bottom - 2.2 * label_size, "center")
            )
        for value, y in yt:
            layer = layer.alpha_overlay(
                label(f"{value:g}", left - 6, y - 0.75 * label_size, "right")
            )
        if x_label:
            y = (
                bottom - 2.2 * label_size
                if axis_y == bottom
                else axis_y - 0.75 * label_size
            )
            layer = layer.alpha_overlay(label(x_label, right + 4, y, "left"))
        if y_label:
            layer = layer.alpha_overlay(
                label(y_label, axis_x - 6, top - 0.25 * label_size, "right")
            )
        return layer

    def annotate(
        self,
        point: tuple[Any, Any],
        label: str | SDFTexture,
        *,
        offset: tuple[Any, Any] = (8.0, 8.0),
        size: float = 16.0,
        tint="#f4f1e9",
        leader_width: Any = None,
        label_origin: Point = (0.0, 0.0),
        cache: bool = True,
    ) -> ColorTexture:
        """Place a label at a data point plus a canvas-space offset.

        Text uses the destination as its left baseline origin. An SDF label
        translates its ``label_origin`` to the destination. A materialized SDF
        can be a static native graph input while its position and the optional
        leader remain trainable. Use ``cache=False`` for moving annotations.
        """
        from . import sdf

        if isinstance(leader_width, Real) and (
            not math.isfinite(leader_width) or leader_width <= 0
        ):
            raise ValueError("annotation leader_width must be finite and positive")
        anchor = self.point(*point)
        dx, dy = map(_lazy_scalar, offset)
        x, y = anchor[0] + dx, anchor[1] + dy
        if isinstance(label, SDFTexture):
            origin_x, origin_y = label_origin
            glyphs = label.translate((x - origin_x, y - origin_y))
        else:
            glyphs = sdf.text(label, size=size, ox=x, oy=y)
        if not cache:
            glyphs = glyphs.uncached()
        background = _transparent_ink(tint)
        ink = glyphs.fill(tint, background, inflate=0.2, inner=-0.5, outer=0.5)
        if leader_width is None:
            return ink if cache else ink.uncached()

        line = sdf.segment(anchor, (x, y), _lazy_scalar(leader_width) / 2)
        if not cache:
            line = line.uncached()
            ink = ink.uncached()
        leader = line.fill(tint, background)
        layer = (leader if cache else leader.uncached()).alpha_overlay(ink)
        return layer if cache else layer.uncached()
