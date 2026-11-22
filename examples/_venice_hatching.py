"""Find crosshatch segments inside building footprints."""

import math
from itertools import pairwise


def _ring_edges(rings, normal):
    return tuple(
        (
            first,
            second,
            first[0] * normal[0] + first[1] * normal[1],
            second[0] * normal[0] + second[1] * normal[1],
        )
        for ring in rings
        for first, second in pairwise(ring)
    )


def _line_intersections(edges, level, direction):
    intersections = []
    for first, second, first_value, second_value in edges:
        if (first_value <= level < second_value) or (
            second_value <= level < first_value
        ):
            amount = (level - first_value) / (second_value - first_value)
            x = first[0] + amount * (second[0] - first[0])
            y = first[1] + amount * (second[1] - first[1])
            intersections.append(x * direction[0] + y * direction[1])
    intersections.sort()
    return intersections


def _hatch_direction(rings, direction, normal, spacing):
    edges = _ring_edges(rings, normal)
    coordinates = [value for _, _, first, second in edges for value in (first, second)]
    low = math.floor(min(coordinates) / spacing)
    high = math.ceil(max(coordinates) / spacing)
    for index in range(low, high + 1):
        level = index * spacing
        intersections = _line_intersections(edges, level, direction)
        for start, end in zip(intersections[::2], intersections[1::2]):
            first = (
                (start * direction[0] + level * normal[0]) / 2,
                (start * direction[1] + level * normal[1]) / 2,
            )
            second = (
                (end * direction[0] + level * normal[0]) / 2,
                (end * direction[1] + level * normal[1]) / 2,
            )
            if first != second:
                yield first, second


def hatch_segments(rings, spacing):
    """Fill footprint boundaries and preserve holes with two diagonal passes."""
    rings = tuple(ring for ring in rings if len(ring) >= 2)
    if not rings:
        return ()
    return tuple(
        path
        for direction, normal in (((1, -1), (1, 1)), ((1, 1), (1, -1)))
        for path in _hatch_direction(rings, direction, normal, spacing)
    )
