"""Scale fitting and panel rendering for the data chart grammar."""

from __future__ import annotations


def build_layers(spec, box):
    from ..core.expressions import _lazy_scalar
    from ._plot_data import (
        _MUTED,
        PlotLayers,
        _chart_box,
        _compose,
        _legend,
        _panels,
        _prepare,
        _text,
    )
    from ._plot_geometries import X_CHANNELS, Y_CHANNELS
    from ._plot_scales import PositionScale

    box = _chart_box(box)
    prepared = _prepare_layers(spec, _prepare)
    panels = _panel_keys(prepared)
    if not panels:
        raise ValueError("chart requires at least one layer with data")
    x_scale, y_scale = _fit_scales(spec, prepared, PositionScale, X_CHANNELS, Y_CHANNELS)
    colors = _fit_colors(spec, prepared)
    legend_labels, legend_swatches, legend_rows = _legend(box, colors)
    chrome, marks = _render_panels(
        spec,
        box,
        panels,
        prepared,
        x_scale,
        y_scale,
        colors,
        legend_rows,
        _panels,
        _text,
        _lazy_scalar,
        X_CHANNELS,
        Y_CHANNELS,
        _MUTED,
    )
    chrome.extend(spec._headings(box))
    chrome.extend(legend_labels)
    marks.extend(legend_swatches)
    return PlotLayers(_compose(chrome), _compose(marks))


def _prepare_layers(spec, prepare):
    return [
        prepare(layer, dict(spec.mapping), spec.data, spec._facets)
        for layer in spec._layers
    ]


def _panel_keys(prepared):
    return tuple(dict.fromkeys(panel for layer in prepared for panel in layer))


def _fit_scales(spec, prepared, scale_type, x_channels, y_channels):
    groups = [
        group for layer in prepared for panel in layer.values() for group in panel
    ]
    rows = tuple(row for group in groups for row in group.rows)
    x_scale = scale_type.fit(
        (row[key] for row in rows for key in x_channels if key in row),
        spec.xlim,
        axis="x",
    )
    x_scale = _fit_padded_x_scale(spec, prepared, rows, x_scale, scale_type, x_channels)
    y_scale = scale_type.fit(
        (row[key] for row in rows for key in y_channels if key in row),
        spec.ylim,
        axis="y",
        include_zero=any(layer.geometry.include_zero for layer in spec._layers),
    )
    return x_scale, y_scale


def _fit_padded_x_scale(spec, prepared, rows, x_scale, scale_type, x_channels):
    if spec.xlim is not None or x_scale.categories:
        return x_scale
    values = (
        row[key] + group.offset + edge
        for layer, panel_data in zip(spec._layers, prepared)
        for cell in panel_data.values()
        for group in cell
        for row in group.rows
        for key in x_channels
        if key in row
        for edge in _x_padding(layer, key)
    )
    return scale_type.fit(values, None, axis="x")


def _x_padding(layer, key):
    if key == "x" and layer.geometry.x_padding:
        return -layer.geometry.x_padding, layer.geometry.x_padding
    return (0.0,)


def _fit_colors(spec, prepared):
    color_keys = tuple(dict.fromkeys(
        group.color
        for layer, data in zip(spec._layers, prepared)
        if layer.tint is None
        for cell in data.values()
        for group in cell
        if group.color is not None
    ))
    return {key: spec.palette[index % len(spec.palette)] for index, key in enumerate(color_keys)}


def _render_panels(
    spec, box, panels, prepared, x_scale, y_scale, colors, legend_rows,
    panel_layout, text, lazy_scalar, x_channels, y_channels, muted,
):
    chrome, marks = [], []
    plots = panel_layout(
        box, panels, spec._facets, x_scale, y_scale, spec._labels, legend_rows
    )
    for panel, plot in plots:
        _draw_panel_chrome(spec, panel, plot, x_scale, y_scale, chrome, text, muted)
        _draw_panel_marks(
            spec, panel, plot, prepared, x_scale, y_scale, colors, marks,
            lazy_scalar, x_channels, y_channels,
        )
    return chrome, marks


def _draw_panel_chrome(spec, panel, plot, x_scale, y_scale, chrome, text, muted):
    left, bottom, _, height = plot.box
    axis_y = (
        0
        if y_scale.limits[0] <= 0 <= y_scale.limits[1] and not y_scale.categories
        else y_scale.limits[0]
    )
    chrome.append(
        plot.axes(
            x_ticks=x_scale.ticks(),
            y_ticks=y_scale.ticks(),
            x_format=x_scale.label,
            y_format=y_scale.label,
            x_axis_at=axis_y,
            y_axis_at=x_scale.limits[0],
            axis_color="#748ba1",
            grid_color="#34495d",
            label_color=muted,
            label_size=13,
        )
    )
    if spec._facets:
        chrome.append(text(str(panel), left, bottom + height + 12, 15))


def _draw_panel_marks(
    spec, panel, plot, prepared, x_scale, y_scale, colors, marks,
    lazy_scalar, x_channels, y_channels,
):
    for layer, panel_groups in zip(spec._layers, prepared):
        for group in panel_groups.get(panel, ()):
            tint = layer.tint if layer.tint is not None else colors.get(group.color, spec.palette[0])
            resolved = _resolve_rows(group.rows, group.offset, x_scale, y_scale,
                                     lazy_scalar, x_channels, y_channels)
            mark = layer.geometry.draw(plot, resolved, tint)
            if mark is not None:
                marks.append(mark if layer.alpha is None else mark.transparency(layer.alpha))


def _resolve_rows(rows, offset, x_scale, y_scale, lazy_scalar, x_channels, y_channels):
    return tuple(
        {
            **{
                key: lazy_scalar(x_scale.encode(row[key])) + offset
                for key in x_channels
                if key in row
            },
            **{key: y_scale.encode(row[key]) for key in y_channels if key in row},
        }
        for row in rows
    )
