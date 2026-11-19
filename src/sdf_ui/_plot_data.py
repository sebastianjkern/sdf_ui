"""Compose data charts from column mappings, geometry layers, and shared scales."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from itertools import chain
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

from ._plot_geometries import (
    X_CHANNELS,
    Y_CHANNELS,
    Geometry,
    columns,
    lines,
    points,
    ranges,
    rectangles,
    ribbons,
)
from ._plot_scales import PositionScale
from ._plot_statistics import Histogram
from ._plot_viewport import Box, Plot, Point
from .core.color import ColorTexture
from .core.expressions import _lazy_scalar
from .text import measure_text

Row = Mapping[str, Any]
MappingSpec = Mapping[str, str]
_CHANNELS = {*X_CHANNELS, *Y_CHANNELS, "color", "group"}
_PALETTE = ("#66e0c5", "#ffc880", "#92baff", "#ffa8cb", "#c4b2ff", "#aed787")
_FOREGROUND = "#ecf1f9"
_MUTED = "#a0b3c9"
_TRANSPARENT = (0.0, 0.0, 0.0, 0.0)


def aes(**columns: str) -> MappingSpec:
    """Map positions, bounds, color, or group to named data columns."""
    return _mapping(columns)


def _mapping(mapping):
    for channel, name in mapping.items():
        if channel not in _CHANNELS:
            raise ValueError(f"unknown aesthetic {channel!r}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{channel} requires a column name")
    return MappingProxyType(dict(mapping))


def _rows(data):
    return tuple(MappingProxyType(dict(row)) for row in data)


@dataclass(frozen=True)
class Layer:
    """One geometry, its mapping overrides, and optional replacement data.

    Use a custom Geometry to add marks that consume the shared chart scales.
    Position offsets and grouping are resolved before the geometry draws.
    """

    geometry: Geometry
    mapping: MappingSpec = field(default_factory=dict)
    data: Iterable[Row] | None = None
    tint: Any = None
    dodge: float = 0.0
    count: bool = False
    alpha: Any = None
    _histogram: Histogram | None = field(default=None, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "mapping", _mapping(self.mapping))
        if self.data is not None:
            object.__setattr__(self, "data", _rows(self.data))
        if not math.isfinite(self.dodge) or self.dodge < 0:
            raise ValueError("dodge must be finite and nonnegative")
        if isinstance(self.alpha, Real) and not 0 <= self.alpha <= 1:
            raise ValueError("layer alpha must be between zero and one")


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


@dataclass(frozen=True)
class _Labels:
    title: str | None
    x: str | None
    y: str | None


def labs(
    *, title: str | None = None, x: str | None = None, y: str | None = None
) -> _Labels:
    """Set a title and axis labels. An empty string hides a label."""
    return _Labels(title, x, y)


@dataclass(frozen=True)
class _Facets:
    column: str
    columns: int


def facet_wrap(column: str, *, columns: int = 2) -> _Facets:
    """Create small multiples with shared position and color scales."""
    if not isinstance(column, str) or not column:
        raise ValueError("facets require a column name")
    if isinstance(columns, bool) or not isinstance(columns, Integral) or columns < 1:
        raise ValueError("facet columns must be a positive integer")
    return _Facets(column, columns)


@dataclass(frozen=True)
class _Group:
    rows: tuple[Row, ...]
    color: Any = None
    offset: float = 0.0


def _column(row, name):
    try:
        return row[name]
    except KeyError:
        raise ValueError(f"missing data column {name!r}") from None


def _category(value, channel):
    if isinstance(value, str) or (isinstance(value, Real) and math.isfinite(value)):
        return value
    raise ValueError(f"{channel} values must be strings or finite numbers")


def _prepare(layer, inherited, data, facets):
    mapping = inherited | dict(layer.mapping)
    channels = ("x",) if layer.count or layer._histogram else layer.geometry.channels
    missing = [channel for channel in channels if channel not in mapping]
    if missing:
        raise ValueError(f"geometry requires mappings for {', '.join(missing)}")
    groups = {}
    keys = {}
    for row in data if layer.data is None else layer.data:
        panel = _category(_column(row, facets.column), "facet") if facets else None
        tint_key = (
            _category(_column(row, mapping["color"]), "color")
            if "color" in mapping
            else None
        )
        group_key = (
            _category(_column(row, mapping["group"]), "group")
            if "group" in mapping
            else None
        )
        key = (tint_key, group_key)
        keys.setdefault(key, len(keys))
        groups.setdefault(panel, {}).setdefault(key, []).append(
            {channel: _column(row, mapping[channel]) for channel in channels}
        )
    edges = (
        layer._histogram.fit(
            chain.from_iterable(
                rows for panel in groups.values() for rows in panel.values()
            )
        )
        if layer._histogram
        else ()
    )
    result = {}
    for panel, panel_groups in groups.items():
        prepared = []
        for key, rows in panel_groups.items():
            if layer.count:
                counts = Counter(_category(row["x"], "count x") for row in rows)
                rows = [{"x": x, "y": n} for x, n in counts.items()]
            elif layer._histogram:
                rows = layer._histogram.apply(rows, edges)
            offset = (keys[key] - (len(keys) - 1) / 2) * layer.dodge
            prepared.append(_Group(tuple(rows), key[0], offset))
        result[panel] = prepared
    return result


def _compose(layers):
    from . import color

    return color.clear(_TRANSPARENT).over(*layers)


def _text(label, x, y, size, tint=_FOREGROUND):
    from . import sdf

    return sdf.text(label, ox=x, oy=y, size=size).ink(tint, inflate=0.2)


@dataclass(frozen=True)
class PlotLayers:
    """Static chart furniture and geometry kept as separate scene layers."""

    chrome: ColorTexture
    marks: ColorTexture

    def scene(self) -> ColorTexture:
        return self.chrome.over(self.marks)


@dataclass(frozen=True)
class PlotSpec:
    """An immutable specification for a chart built from rows of named values.

    Plot mappings apply to all layers and can be overridden by each layer.
    Position scales are inferred across layers and facets. Supply explicit
    limits for axes that use lazy coordinates.
    """

    data: Iterable[Row]
    mapping: MappingSpec = field(default_factory=dict)
    xlim: Point | None = None
    ylim: Point | None = None
    palette: tuple[Any, ...] = _PALETTE
    _layers: tuple[Layer, ...] = field(default=(), repr=False)
    _labels: _Labels = field(default_factory=lambda: labs(), repr=False)
    _facets: _Facets | None = field(default=None, repr=False)

    def __post_init__(self):
        if not isinstance(self.data, tuple) or not all(
            isinstance(row, MappingProxyType) for row in self.data
        ):
            object.__setattr__(self, "data", _rows(self.data))
        object.__setattr__(self, "mapping", _mapping(self.mapping))
        object.__setattr__(self, "palette", tuple(self.palette))
        if not self.palette:
            raise ValueError("chart palette cannot be empty")

    def __add__(self, component: Layer | _Labels | _Facets) -> PlotSpec:
        if isinstance(component, Layer):
            return replace(self, _layers=(*self._layers, component))
        if isinstance(component, _Labels):
            return replace(
                self,
                _labels=replace(
                    self._labels,
                    **{
                        key: value
                        for key, value in vars(component).items()
                        if value is not None
                    },
                ),
            )
        if isinstance(component, _Facets):
            return replace(self, _facets=component)
        return NotImplemented

    def scene(self, box: Box) -> ColorTexture:
        """Build the complete transparent chart layer for a canvas box."""
        return self.build(box).scene()

    def build(self, box: Box) -> PlotLayers:
        """Build marks separately so static axes and text can be rendered once."""
        box = _chart_box(box)
        prepared = [
            _prepare(layer, dict(self.mapping), self.data, self._facets)
            for layer in self._layers
        ]
        panels = tuple(dict.fromkeys(panel for layer in prepared for panel in layer))
        if not panels:
            raise ValueError("chart requires at least one layer with data")
        groups = [
            group for layer in prepared for panel in layer.values() for group in panel
        ]
        rows = tuple(row for group in groups for row in group.rows)
        x_scale = PositionScale.fit(
            (row[key] for row in rows for key in X_CHANNELS if key in row),
            self.xlim,
            axis="x",
        )
        # Center positions alone can leave the outer bars clipped.
        if self.xlim is None and not x_scale.categories:
            x_scale = PositionScale.fit(
                (
                    row[key] + group.offset + edge
                    for layer, data in zip(self._layers, prepared)
                    for cell in data.values()
                    for group in cell
                    for row in group.rows
                    for key in X_CHANNELS
                    if key in row
                    for edge in (
                        (-layer.geometry.x_padding, layer.geometry.x_padding)
                        if key == "x" and layer.geometry.x_padding
                        else (0.0,)
                    )
                ),
                None,
                axis="x",
            )
        y_scale = PositionScale.fit(
            (row[key] for row in rows for key in Y_CHANNELS if key in row),
            self.ylim,
            axis="y",
            include_zero=any(layer.geometry.include_zero for layer in self._layers),
        )
        color_keys = tuple(
            dict.fromkeys(
                group.color
                for layer, data in zip(self._layers, prepared)
                if layer.tint is None
                for cell in data.values()
                for group in cell
                if group.color is not None
            )
        )
        colors = {
            key: self.palette[i % len(self.palette)] for i, key in enumerate(color_keys)
        }
        legend_labels, legend_swatches, legend_rows = _legend(box, colors)
        chrome, marks = [], []
        for panel, plot in _panels(
            box, panels, self._facets, x_scale, y_scale, self._labels, legend_rows
        ):
            left, bottom, _, height = plot.box
            chrome.append(
                plot.axes(
                    x_ticks=x_scale.ticks(),
                    y_ticks=y_scale.ticks(),
                    x_format=x_scale.label,
                    y_format=y_scale.label,
                    x_axis_at=0
                    if y_scale.limits[0] <= 0 <= y_scale.limits[1]
                    and not y_scale.categories
                    else y_scale.limits[0],
                    y_axis_at=x_scale.limits[0],
                    axis_color="#748ba1",
                    grid_color="#34495d",
                    label_color=_MUTED,
                    label_size=13,
                )
            )
            if self._facets:
                chrome.append(_text(str(panel), left, bottom + height + 12, 15))
            for layer, panel_groups in zip(self._layers, prepared):
                for group in panel_groups.get(panel, ()):
                    tint = (
                        layer.tint
                        if layer.tint is not None
                        else colors.get(group.color, self.palette[0])
                    )
                    resolved = tuple(
                        {
                            **{
                                key: _lazy_scalar(x_scale.encode(row[key]))
                                + group.offset
                                for key in X_CHANNELS
                                if key in row
                            },
                            **{
                                key: y_scale.encode(row[key])
                                for key in Y_CHANNELS
                                if key in row
                            },
                        }
                        for row in group.rows
                    )
                    mark = layer.geometry.draw(plot, resolved, tint)
                    if mark is not None:
                        marks.append(
                            mark
                            if layer.alpha is None
                            else mark.transparency(layer.alpha)
                        )
        chrome.extend(self._headings(box))
        chrome.extend(legend_labels)
        marks.extend(legend_swatches)
        return PlotLayers(_compose(chrome), _compose(marks))

    def _headings(self, box):
        left, bottom, _, height = box
        top = bottom + height
        title, xlabel, ylabel = self._labels.title, self._labels.x, self._labels.y
        if xlabel is None:
            xlabel = self.mapping.get("x", "")
        if ylabel is None:
            ylabel = (
                self.mapping.get("y", "")
                if not all(layer.count or layer._histogram for layer in self._layers)
                else "Count"
            )
        if title:
            yield _text(title, left + 15, top - 32, 27)
        if ylabel:
            yield _text(ylabel, left + 15, top - (55 if title else 21), 14, _MUTED)
        if xlabel:
            yield _text(xlabel, left + 15, bottom + 8, 14, _MUTED)


def _chart_box(box):
    left, bottom, width, height = map(float, box)
    if not all(map(math.isfinite, (left, bottom, left + width, bottom + height))):
        raise ValueError("chart box must be finite")
    if width <= 0 or height <= 0:
        raise ValueError("chart box dimensions must be positive")
    return left, bottom, width, height


def _legend(box, colors):
    from . import sdf

    left, bottom, width, _ = box
    start, end = left + 64, left + width - 24
    labels, swatches = [], []
    x, row = start, 0
    for key, tint in colors.items():
        label = str(key)
        item_width = max(50, measure_text(label, size=14)["width"] + 28)
        if item_width > end - start:
            raise ValueError(f"chart box is too narrow for legend label {label!r}")
        if x + item_width > end:
            x, row = start, row + 1
        y = bottom + 42 + row * 20
        swatches.append(sdf.circle((x, y + 4), 4).ink(tint))
        labels.append(_text(label, x + 10, y, 14))
        x += item_width
    return labels, swatches, row + 1 if labels else 0


def _panels(box, panels, facets, x_scale, y_scale, labels, legend_rows):
    left, bottom, width, height = box
    columns = min(facets.columns, len(panels)) if facets else 1
    rows = (len(panels) + columns - 1) // columns
    gap_x, gap_y = 78, 62
    inset_left, inset_right = 64, 24
    inset_top = (75 if labels.title else 39) + (20 if facets else 0)
    inset_bottom = 59 + (8 + 20 * legend_rows if legend_rows else 0)
    inner_width = (width - inset_left - inset_right - gap_x * (columns - 1)) / columns
    inner_height = (height - inset_top - inset_bottom - gap_y * (rows - 1)) / rows
    if inner_width <= 0 or inner_height <= 0:
        raise ValueError("chart box is too small for its panels and labels")
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        x = left + inset_left + column * (inner_width + gap_x)
        y = bottom + height - inset_top - inner_height - row * (inner_height + gap_y)
        yield (
            panel,
            Plot((x, y, inner_width, inner_height), x_scale.limits, y_scale.limits),
        )
