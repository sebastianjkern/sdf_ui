"""Resolve row mappings, groups, facets, and precomputed statistics."""

from __future__ import annotations

from collections import Counter
from itertools import chain


def prepare_layer(layer, inherited, data, facets):
    from ._plot_data import _category, _column, _Group

    mapping = inherited | dict(layer.mapping)
    channels = ("x",) if layer.count or layer._histogram else layer.geometry.channels
    _validate_channels(channels, mapping)
    groups, keys = _collect_rows(layer, mapping, channels, data, facets, _category, _column)
    edges = _fit_histogram(layer, groups)
    return _materialize_groups(layer, groups, keys, edges, _Group, _category)


def _validate_channels(channels, mapping):
    missing = [channel for channel in channels if channel not in mapping]
    if missing:
        raise ValueError(f"geometry requires mappings for {', '.join(missing)}")


def _collect_rows(layer, mapping, channels, data, facets, category, column):
    groups = {}
    keys = {}
    for row in data if layer.data is None else layer.data:
        panel = category(column(row, facets.column), "facet") if facets else None
        key = _group_key(row, mapping, category, column)
        keys.setdefault(key, len(keys))
        groups.setdefault(panel, {}).setdefault(key, []).append(
            {channel: column(row, mapping[channel]) for channel in channels}
        )
    return groups, keys


def _group_key(row, mapping, category, column):
    tint = category(column(row, mapping["color"]), "color") if "color" in mapping else None
    group = category(column(row, mapping["group"]), "group") if "group" in mapping else None
    return tint, group


def _fit_histogram(layer, groups):
    if layer._histogram is None:
        return ()
    rows = chain.from_iterable(
        group_rows for panel in groups.values() for group_rows in panel.values()
    )
    return layer._histogram.fit(rows)


def _materialize_groups(layer, groups, keys, edges, group_type, category):
    prepared_panels = {}
    group_count = len(keys)
    for panel, panel_groups in groups.items():
        prepared_panels[panel] = [
            group_type(
                tuple(_materialize_rows(layer, rows, edges, category)),
                key[0],
                (keys[key] - (group_count - 1) / 2) * layer.dodge,
            )
            for key, rows in panel_groups.items()
        ]
    return prepared_panels


def _materialize_rows(layer, rows, edges, category):
    if layer.count:
        counts = Counter(category(row["x"], "count x") for row in rows)
        return [{"x": value, "y": count} for value, count in counts.items()]
    if layer._histogram:
        return layer._histogram.apply(rows, edges)
    return rows
