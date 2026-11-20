"""Cartesian plots built from composable SDF UI layers."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real
from typing import TYPE_CHECKING, Any

from ..core.color import ColorTexture
from ..core.expressions import (
    _lazy_scalar,
    _scalar_maximum,
    _scalar_minimum,
    _select_positive,
)
from ..core.sdf import SDFTexture
from ._plot_sampling import Point, _ordered_range, _sample_path

if TYPE_CHECKING:
    from ._plot_data import PlotSpec

Box = tuple[float, float, float, float]
DataPoint = tuple[Any, Any]
_LEADER_INNER = -3 / 2
_LEADER_OUTER = 0.0


def _nonnegative_literal(value, name):
    if isinstance(value, Real) and (not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")


def _combine_marks(marks, name):
    """Build a balanced union and let the renderer recycle internal textures."""
    nodes = list(marks)
    if not nodes:
        raise ValueError(f"{name} requires at least one mark")
    if len(nodes) == 1:
        return nodes[0]
    nodes = [node.uncached() for node in nodes]
    while len(nodes) > 2:
        nodes = [
            nodes[i].union(nodes[i + 1]).uncached() if i + 1 < len(nodes) else nodes[i]
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0].union(nodes[1])


def _auto_ticks(limits):
    """Choose about six intervals using steps of 1, 2 or 5 times a power of ten."""
    low, high = limits
    rough = (high - low) / 6
    if rough <= 0:
        return limits
    exponent = math.floor(math.log10(rough))
    power = 10.0**exponent
    if power == 0:
        return limits
    step = next(value * power for value in (1, 2, 5, 10) if value * power >= rough)
    return tuple(
        tick
        for index in range(math.floor(low / step), math.ceil(high / step) + 1)
        if low <= (tick := round(index * step, max(0, -exponent))) <= high
    )


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

    @staticmethod
    def from_data(
        data: Iterable[Mapping[str, Any]],
        mapping: Mapping[str, str] | None = None,
        *,
        xlim: Point | None = None,
        ylim: Point | None = None,
        palette: Iterable[Any] | None = None,
    ) -> PlotSpec:
        """Start a layered plot with optional data mappings and fixed limits.

        Add geometry from :mod:`sdf_ui.plot` with ``+``, then call
        ``.scene(box)`` to place the plot on the canvas.
        """
        from ._plot_data import PlotSpec

        if palette is None:
            return PlotSpec(data, mapping or {}, xlim=xlim, ylim=ylim)
        return PlotSpec(data, mapping or {}, xlim=xlim, ylim=ylim, palette=tuple(palette))

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

    def data_point(self, x: Any, y: Any) -> DataPoint:
        """Map a canvas point back to data coordinates, including outside the box."""
        left, bottom, width, height = self.box
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        x, y = _lazy_scalar(x), _lazy_scalar(y)
        return (
            x_min + (x - left) / width * (x_max - x_min),
            y_min + (y - bottom) / height * (y_max - y_min),
        )

    def clip(self, shape: SDFTexture) -> SDFTexture:
        """Intersect a filled shape with the viewport. Apply ink after clipping."""
        from .. import sdf

        left, bottom, width, height = self.box
        viewport = sdf.rect_from_size(
            (left + width / 2, bottom + height / 2), (width, height)
        )
        return shape.intersection(viewport)

    def line(
        self, points: Iterable[DataPoint | None], *, radius: Any = 0.0
    ) -> SDFTexture:
        """Connect adjacent data points. Use ``None`` to start a new run.

        Radii are in canvas units. A zero radius gives a centerline for
        ``.stroke()``. For clipped ink, use a positive radius and pass the band
        to ``clip()`` before coloring. Coordinates and radius can be lazy.
        Supply at least one pair of adjacent points.
        """
        from .. import sdf

        _nonnegative_literal(radius, "line radius")
        mapped = (None if point is None else self.point(*point) for point in points)
        segments = (
            sdf.segment(a, b, radius)
            for a, b in pairwise(mapped)
            if a is not None and b is not None
        )
        return _combine_marks(segments, "line")

    def scatter(self, points: Iterable[DataPoint], *, radius: Any = 4.0) -> SDFTexture:
        """Draw one or more circles with a canvas-space radius and live positions."""
        from .. import sdf

        _nonnegative_literal(radius, "scatter radius")
        return _combine_marks(
            (sdf.circle(self.point(*point), radius) for point in points), "scatter"
        )

    def bars(
        self, points: Iterable[DataPoint], *, width: Any = 0.8, baseline: Any = 0.0
    ) -> SDFTexture:
        """Draw vertical bars centered on each x, with width in x data units.

        Values can extend above or below the shared baseline. Coordinates,
        width and baseline can all be lazy expressions. A value on the
        baseline gives a zero-height rectangle.
        """
        from .. import sdf

        _nonnegative_literal(width, "bar width")
        canvas_width = self.box[2]
        pixel_width = (
            _lazy_scalar(width) / (self.x_range[1] - self.x_range[0]) * canvas_width
        )
        base_y = self.point(self.x_range[0], baseline)[1]

        def marks():
            for x, y in points:
                px, py = self.point(x, y)
                yield sdf.rect_from_size(
                    (px, base_y + (py - base_y) / 2),
                    (pixel_width, abs(py - base_y)),
                )

        return _combine_marks(marks(), "bars")

    def rectangles(self, bounds: Iterable[tuple[Any, Any, Any, Any]]) -> SDFTexture:
        """Draw boxes from (xmin, xmax, ymin, ymax) data coordinates.

        Corner coordinates may be lazy and can be supplied in either order
        on each axis. Apply ``clip()`` before coloring to crop to the viewport.
        """
        from .. import sdf

        def marks():
            for xmin, xmax, ymin, ymax in bounds:
                x0, y0 = self.point(xmin, ymin)
                x1, y1 = self.point(xmax, ymax)
                yield sdf.rect_from_size(
                    (x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2),
                    (abs(x1 - x0), abs(y1 - y0)),
                )

        return _combine_marks(marks(), "rectangles")

    def ribbon(self, points: Iterable[tuple[Any, Any, Any] | None]) -> SDFTexture:
        """Fill between lower and upper values in each (x, low, high) sample.

        Adjacent samples form a linear span, with ``None`` separating runs.
        Each coordinate may be lazy. Keep x ordered within a run and supply
        at least two adjacent samples. Paint with ``ink()`` for best results.
        """
        from .. import sdf

        def marks():
            samples = tuple(points)
            for index, (a, b) in enumerate(pairwise(samples)):
                if a is None or b is None:
                    continue
                yield _ribbon_span(sdf, self, samples, index, a, b)

        return _combine_marks(marks(), "ribbon")

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
        from .. import sdf

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
        from .. import sdf

        left, bottom, width, height = self.box
        points = _sample_path(
            function,
            domain,
            self.point,
            (left, bottom),
            (left + width, bottom + height),
            tolerance,
            samples,
            breaks,
        )
        return sdf.polyline(points, radius)

    def axes(
        self,
        x_ticks: Iterable[float] | None = (),
        y_ticks: Iterable[float] | None = (),
        *,
        axis_color="#a4adbc",
        grid_color=(0.35, 0.40, 0.49, 0.45),
        label_color=None,
        axis_width: Any = 2.0,
        grid_width: Any = 1.0,
        label_size: float = 15.0,
        x_label: str = "",
        y_label: str = "",
        x_format: Callable[[float], str] | None = None,
        y_format: Callable[[float], str] | None = None,
        x_axis_at: float | None = None,
        y_axis_at: float | None = None,
    ) -> ColorTexture:
        """Draw axes, grid and labels. Pass ``None`` for automatic ticks.

        Explicit tick values can be formatted independently for each axis.
        The empty iterable keeps the axis free of grid lines and tick labels.
        Set ``x_axis_at`` to a y data value or ``y_axis_at`` to an x data value
        to change where the axes cross. By default each axis passes through
        zero when visible and follows the lower edge of its range otherwise.
        """
        from ._plot_axes import draw_axes

        return draw_axes(
            self,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            axis_color=axis_color,
            grid_color=grid_color,
            label_color=label_color,
            axis_width=axis_width,
            grid_width=grid_width,
            label_size=label_size,
            x_label=x_label,
            y_label=y_label,
            x_format=x_format,
            y_format=y_format,
            x_axis_at=x_axis_at,
            y_axis_at=y_axis_at,
        )

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
        from .. import sdf

        if isinstance(leader_width, Real) and (
            not math.isfinite(leader_width) or leader_width <= 0
        ):
            raise ValueError("annotation leader_width must be finite and positive")
        anchor = self.point(*point)
        dx, dy = map(_lazy_scalar, offset)
        x, y = anchor[0] + dx, anchor[1] + dy
        return _annotation_layer(
            sdf, anchor, (x, y), label, label_origin, size, tint, leader_width, cache
        )


def _annotation_layer(sdf, anchor, target, label, origin, size, tint, leader_width, cache):
    glyphs = _annotation_glyphs(sdf, target, label, origin, size)
    glyphs = glyphs if cache else glyphs.uncached()
    ink = glyphs.ink(tint, inflate=0.2)
    if leader_width is None:
        return ink if cache else ink.uncached()
    line = sdf.segment(anchor, target, _lazy_scalar(leader_width) / 2)
    return _annotation_composite(line, ink, tint, cache)


def _annotation_glyphs(sdf, target, label, origin, size):
    x, y = target
    if isinstance(label, SDFTexture):
        origin_x, origin_y = origin
        return label.translate((x - origin_x, y - origin_y))
    return sdf.text(label, size=size, ox=x, oy=y)


def _annotation_composite(line, ink, tint, cache):
    line = line if cache else line.uncached()
    ink = ink if cache else ink.uncached()
    leader = line.ink(tint, inner=_LEADER_INNER, outer=_LEADER_OUTER)
    layer = (leader if cache else leader.uncached()).alpha_overlay(ink)
    return layer if cache else layer.uncached()


def _ribbon_span(sdf, plot, samples, index, a, b):
    xa, low_a = plot.point(a[0], a[1])
    _, high_a = plot.point(a[0], a[2])
    xb, low_b = plot.point(b[0], b[1])
    _, high_b = plot.point(b[0], b[2])
    dx = xb - xa
    sign = _select_positive(dx, 1.0, -1.0)
    pad_a = _ribbon_padding(samples, index, previous=True)
    pad_b = _ribbon_padding(samples, index, previous=False)
    x0 = _scalar_minimum(xa, xb) - _select_positive(dx, pad_a, pad_b)
    x1 = _scalar_maximum(xa, xb) + _select_positive(dx, pad_b, pad_a)
    bottom = sdf.half_plane((xa, low_a), (sign * (low_b - low_a), -abs(dx)))
    top = sdf.half_plane((xa, high_a), (sign * (high_a - high_b), abs(dx)))
    left = sdf.half_plane((x0, 0.0), (-1, 0))
    right = sdf.half_plane((x1, 0.0), (1, 0))
    return (
        bottom.uncached()
        .intersection(top.uncached())
        .uncached()
        .intersection(left.uncached())
        .uncached()
        .intersection(right.uncached())
    )


def _ribbon_padding(samples, index, *, previous):
    neighbor_index = index - 1 if previous else index + 2
    in_bounds = neighbor_index >= 0 if previous else neighbor_index < len(samples)
    return 1.0 if in_bounds and samples[neighbor_index] is not None else 0.0
