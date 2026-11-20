"""Compose data charts from column mappings, geometry layers, and shared scales."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

from ._plot_geometries import (
    X_CHANNELS,
    Y_CHANNELS,
    Geometry,
)
from ._plot_statistics import Histogram
from ._plot_viewport import Box, Plot, Point
from .core.color import ColorTexture
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
    from ._plot_data_prepare import prepare_layer

    return prepare_layer(layer, inherited, data, facets)


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
        from ._plot_data_render import build_layers

        return build_layers(self, box)

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
