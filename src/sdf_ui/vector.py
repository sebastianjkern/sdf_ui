"""Vector geometry and drawing descriptions for sampled scalar fields."""

import math
from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]
Polyline = tuple[Point, ...]
SADDLE_CENTER_WEIGHT = 0.25
GEOMETRY_EPSILON = 1e-8
HATCH_LINE_LIMIT = 100_000


@dataclass(frozen=True)
class VectorPath:
    """A pen centerline in canvas units with a top-left origin and Y down."""

    points: tuple[Point, ...]
    color: str
    width: float
    pen: str | None = None


@dataclass(frozen=True)
class VectorDrawing:
    """A canvas viewport and paths in their authored drawing order."""

    size: tuple[int, int]
    paths: tuple[VectorPath, ...]


def _field_array(field):
    if hasattr(field, "numpy"):
        field = field.numpy()
    array = np.asarray(field, dtype=float)
    if array.ndim != 2:
        raise ValueError("field must be a two-dimensional scalar array")
    return array


def _edge_crossings(values, x, y, level):
    edges = (
        ((x, y), (x + 1, y)),
        ((x + 1, y), (x + 1, y + 1)),
        ((x + 1, y + 1), (x, y + 1)),
        ((x, y + 1), (x, y)),
    )
    crossed = []
    for index, (start, end) in enumerate(edges):
        first, second = float(values[index]), float(values[(index + 1) % 4])
        if not (math.isfinite(first) and math.isfinite(second)) or (first < level) == (
            second < level
        ):
            continue
        amount = (level - first) / (second - first)
        if amount is not None:
            point = (
                start[0] + amount * (end[0] - start[0]),
                start[1] + amount * (end[1] - start[1]),
            )
            crossed.append((index, point))
    return crossed


def _cell_segments(field, x, y, level):
    values = (field[y, x], field[y, x + 1], field[y + 1, x + 1], field[y + 1, x])
    if not all(math.isfinite(float(value)) for value in values):
        return ()
    crossed = _edge_crossings(values, x, y, level)
    if len(crossed) == 2:
        return ((crossed[0][1], crossed[1][1]),)
    if len(crossed) == 4:
        by_edge = dict(crossed)
        center_inside = (
            sum(float(value) for value in values) * SADDLE_CENTER_WEIGHT < level
        )
        pairs = (
            ((0, 1), (2, 3))
            if (float(values[0]) < level) == center_inside
            else ((0, 3), (1, 2))
        )
        return tuple((by_edge[first], by_edge[second]) for first, second in pairs)
    return ()


def _segments(field, level):
    height, width = field.shape
    for y in range(height - 1):
        for x in range(width - 1):
            yield from _cell_segments(field, x, y, level)


def _contour_graph(segments):
    points = {}
    edges = []
    for first, second in segments:
        a = (round(first[0], 10), round(first[1], 10))
        b = (round(second[0], 10), round(second[1], 10))
        if a == b:
            continue
        ia = points.setdefault(a, len(points))
        ib = points.setdefault(b, len(points))
        edges.append((ia, ib))
    coordinates = tuple(points)
    adjacency = [[] for _ in coordinates]
    for index, (a, b) in enumerate(edges):
        adjacency[a].append(index)
        adjacency[b].append(index)
    return coordinates, edges, adjacency


def _walk_contour(start, edge_index, coordinates, edges, adjacency, used):
    chain = [start]
    current = start
    edge = edge_index
    while edge not in used:
        used.add(edge)
        a, b = edges[edge]
        current = b if current == a else a
        chain.append(current)
        choices = [
            candidate for candidate in adjacency[current] if candidate not in used
        ]
        if len(adjacency[current]) != 2 or not choices:
            break
        edge = choices[0]
    return tuple(coordinates[i] for i in chain)


def _stitch(segments):
    coordinates, edges, adjacency = _contour_graph(segments)
    if not edges:
        return ()
    used = set()
    lines = []
    starts = sorted(
        (i for i, adjacent in enumerate(adjacency) if len(adjacent) != 2),
        key=lambda i: coordinates[i],
    )
    for start in starts:
        for edge in adjacency[start]:
            if edge not in used:
                lines.append(
                    _walk_contour(start, edge, coordinates, edges, adjacency, used)
                )
    for edge_index, (start, _) in enumerate(edges):
        if edge_index not in used:
            lines.append(
                _walk_contour(start, edge_index, coordinates, edges, adjacency, used)
            )
    return tuple(lines)


def contour_lines(field, *, level):
    """Return interpolated contour polylines; non-finite cells are skipped."""
    array = _field_array(field)
    if not math.isfinite(float(level)):
        raise ValueError("level must be finite")
    return _stitch(tuple(_segments(array, float(level))))


def _sample(array, x, y):
    height, width = array.shape
    x = min(max(x, 0.0), width - 1.0)
    y = min(max(y, 0.0), height - 1.0)
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    values = (array[y0, x0], array[y0, x1], array[y1, x1], array[y1, x0])
    if not all(math.isfinite(float(value)) for value in values):
        return math.nan
    fx, fy = x - x0, y - y0
    return (1 - fy) * ((1 - fx) * values[0] + fx * values[1]) + fy * (
        (1 - fx) * values[3] + fx * values[2]
    )


def _line_bounds(corners, origin, direction, normal):
    hits = []
    for index, first in enumerate(corners):
        second = corners[(index + 1) % len(corners)]
        a = _dot_delta(first, origin, normal)
        b = _dot_delta(second, origin, normal)
        if a == b == 0.0:
            hits.extend(
                (
                    _dot_delta(first, origin, direction),
                    _dot_delta(second, origin, direction),
                )
            )
        elif (a <= 0 <= b) or (b <= 0 <= a):
            if a != b:
                fraction = -a / (b - a)
                point = (
                    first[0] + fraction * (second[0] - first[0]),
                    first[1] + fraction * (second[1] - first[1]),
                )
                hits.append(_dot_delta(point, origin, direction))
    return (min(hits), max(hits)) if len(hits) >= 2 else None


def _dot_delta(point, origin, vector):
    return (point[0] - origin[0]) * vector[0] + (point[1] - origin[1]) * vector[1]


def _contour_cuts(contours, origin, direction, normal, low, high):
    cuts = [low, high]
    for a, b in contours:
        ta, tb = _dot_delta(a, origin, direction), _dot_delta(b, origin, direction)
        na, nb = _dot_delta(a, origin, normal), _dot_delta(b, origin, normal)
        if na == nb == 0.0:
            cuts.extend((ta, tb))
        elif (na <= 0 <= nb) or (nb <= 0 <= na):
            if na != nb:
                cuts.append(ta - na * (tb - ta) / (nb - na))
    cuts = sorted(min(max(value, low), high) for value in cuts)
    unique = []
    for value in cuts:
        if not unique or abs(value - unique[-1]) > GEOMETRY_EPSILON:
            unique.append(value)
    return unique


def _hatch_interval(array, origin, direction, start, end, level):
    middle = (start + end) * 0.5
    x = origin[0] + direction[0] * middle
    y = origin[1] + direction[1] * middle
    if _sample(array, x, y) >= level:
        return None
    return (
        (origin[0] + direction[0] * start, origin[1] + direction[1] * start),
        (origin[0] + direction[0] * end, origin[1] + direction[1] * end),
    )


def _hatch_parameters(level, angle, spacing, phase, origin):
    level, angle, spacing, phase = map(float, (level, angle, spacing, phase))
    ox, oy = map(float, origin)
    if not all(
        math.isfinite(value) for value in (level, angle, spacing, phase, ox, oy)
    ):
        raise ValueError("level, angle, spacing, phase, and origin must be finite")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    return level, angle, spacing, phase % spacing, ox, oy


def _hatch_indices(corners, origin, normal, spacing, phase):
    projections = [_dot_delta(point, origin, normal) for point in corners]
    first = math.ceil((min(projections) - phase) / spacing)
    last = math.floor((max(projections) - phase) / spacing)
    if last - first + 1 > HATCH_LINE_LIMIT:
        raise ValueError(f"spacing produces more than {HATCH_LINE_LIMIT} hatch lines")
    return range(first, last + 1)


def _hatch_at_offset(
    array, contours, corners, origin, direction, normal, offset, level
):
    line_origin = (origin[0] + normal[0] * offset, origin[1] + normal[1] * offset)
    bounds = _line_bounds(corners, line_origin, direction, normal)
    if bounds is None:
        return ()
    low, high = bounds
    cuts = _contour_cuts(contours, line_origin, direction, normal, low, high)
    intervals = []
    for start, end in zip(cuts, cuts[1:]):
        if end - start > GEOMETRY_EPSILON:
            clipped = _hatch_interval(array, line_origin, direction, start, end, level)
            if clipped is not None:
                intervals.append(clipped)
    return intervals


def hatch_lines(field, *, level, angle, spacing, phase=0.0, origin=(0.0, 0.0)):
    """Clip parallel hatch lines to the region where ``field < level``."""
    array = _field_array(field)
    level, angle, spacing, phase, ox, oy = _hatch_parameters(
        level, angle, spacing, phase, origin
    )
    height, width = array.shape
    if not height or not width:
        return ()
    direction = (math.cos(angle), math.sin(angle))
    normal = (-math.sin(angle), math.cos(angle))
    corners = (
        (-0.5, -0.5),
        (width - 0.5, -0.5),
        (width - 0.5, height - 0.5),
        (-0.5, height - 0.5),
    )
    origin_point = (ox, oy)
    indices = _hatch_indices(corners, origin_point, normal, spacing, phase)
    padded = np.pad(array, 1, mode="edge")
    contours = tuple(
        ((a[0] - 1, a[1] - 1), (b[0] - 1, b[1] - 1))
        for a, b in _segments(padded, level)
    )
    output = []
    for index in indices:
        output.extend(
            _hatch_at_offset(
                array,
                contours,
                corners,
                origin_point,
                direction,
                normal,
                index * spacing + phase,
                level,
            )
        )
    return tuple(output)
