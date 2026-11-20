"""Axis, grid, and tick label construction for Cartesian plots."""

from __future__ import annotations

from ._plot_sampling import _path_points

_LABEL_HEIGHT = 3 / 2
_LABEL_GAP = 2.2
_LABEL_CENTER = 0.75
_LABEL_TOP_INSET = 0.25


def draw_axes(
    plot,
    *,
    x_ticks,
    y_ticks,
    axis_color,
    grid_color,
    label_color,
    axis_width,
    grid_width,
    label_size,
    x_label,
    y_label,
    x_format,
    y_format,
    x_axis_at,
    y_axis_at,
):
    from .. import color, sdf
    from ._plot_viewport import _auto_ticks

    xmin, xmax = plot.x_range
    ymin, ymax = plot.y_range
    left, bottom, width, height = plot.box
    right, top = left + width, bottom + height
    x_axis, y_axis = _axis_values(plot, x_axis_at, y_axis_at)
    axis_x, axis_y = plot.point(y_axis, x_axis)
    x_ticks = _auto_ticks(plot.x_range) if x_ticks is None else x_ticks
    y_ticks = _auto_ticks(plot.y_range) if y_ticks is None else y_ticks
    xt, yt = _tick_positions(plot, x_ticks, y_ticks, xmin, xmax, ymin, ymax)
    layers = _axis_lines(
        sdf, xt, yt, x_axis, y_axis, axis_x, axis_y, left, bottom, right, top,
        axis_color, grid_color, axis_width, grid_width,
    )
    tint = axis_color if label_color is None else label_color
    layers.extend(_tick_labels(
        sdf, xt, yt, plot.box, axis_x, axis_y, label_size, tint, x_format, y_format,
        x_label, y_label,
    ))
    return color.clear((0.0, 0.0, 0.0, 0.0)).over(*layers)


def _axis_values(plot, x_axis_at, y_axis_at):
    xmin, xmax = plot.x_range
    ymin, ymax = plot.y_range
    x_axis = (0.0 if ymin <= 0 <= ymax else ymin) if x_axis_at is None else x_axis_at
    y_axis = (0.0 if xmin <= 0 <= xmax else xmin) if y_axis_at is None else y_axis_at
    if not ymin <= x_axis <= ymax:
        raise ValueError("x_axis_at must be within y_range")
    if not xmin <= y_axis <= xmax:
        raise ValueError("y_axis_at must be within x_range")
    return x_axis, y_axis


def _tick_positions(plot, x_ticks, y_ticks, xmin, xmax, ymin, ymax):
    xt = [(value, plot.point(value, ymin)[0]) for value in x_ticks if xmin <= value <= xmax]
    yt = [(value, plot.point(xmin, value)[1]) for value in y_ticks if ymin <= value <= ymax]
    return xt, yt


def _axis_lines(
    sdf, xt, yt, x_axis, y_axis, axis_x, axis_y, left, bottom, right, top,
    axis_color, grid_color, axis_width, grid_width,
):
    grid = _path_points(
        [((x, bottom), (x, top)) for value, x in xt if value != y_axis]
        + [((left, y), (right, y)) for value, y in yt if value != x_axis]
    )
    layers = []
    if grid:
        layers.append(sdf.polyline(grid).stroke(grid_width, grid_color))
    axes = ((left, axis_y), (right, axis_y), None, (axis_x, bottom), (axis_x, top))
    layers.append(sdf.polyline(axes).stroke(axis_width, axis_color))
    return layers


def _tick_labels(
    sdf, xt, yt, box, axis_x, axis_y, size, tint, x_format, y_format,
    x_label, y_label,
):
    left, bottom, width, height = box
    right, top = left + width, bottom + height
    labels = []
    for value, x in xt:
        value_text = x_format(value) if x_format else f"{value:g}"
        labels.append(_label(sdf, value_text, x, bottom - _LABEL_GAP * size, "center", size, tint))
    for value, y in yt:
        value_text = y_format(value) if y_format else f"{value:g}"
        labels.append(_label(sdf, value_text, left - 6, y - _LABEL_CENTER * size, "right", size, tint))
    if x_label:
        y = bottom - _LABEL_GAP * size if axis_y == bottom else axis_y - _LABEL_CENTER * size
        labels.append(_label(sdf, x_label, right + 4, y, "left", size, tint))
    if y_label:
        y = top - _LABEL_TOP_INSET * size
        labels.append(_label(sdf, y_label, axis_x - 6, y, "right", size, tint))
    return labels


def _label(sdf, value, x, y, align, size, tint):
    width = max(1, len(value)) * size
    if align == "center":
        x -= width / 2
    elif align == "right":
        x -= width
    return sdf.text(
        value,
        size=size,
        box=(x, y, width, _LABEL_HEIGHT * size),
        align=align,
    ).ink(tint, inflate=0.2)
