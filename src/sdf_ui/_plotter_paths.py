from itertools import groupby, pairwise
from math import dist
from operator import itemgetter
from typing import Iterator

import numpy as np

from .vector import Point, Polyline

PenPath = tuple[str | None, Polyline]


def optimize_paths(paths: tuple[PenPath, ...], origin: Point) -> tuple[PenPath, ...]:
    ordered: list[PenPath] = []
    current = origin
    for pen, run in groupby(_effective_pens(paths), key=itemgetter(0)):
        strokes = tuple(points for _, points in run)
        routed = _nearest_paths(strokes, current)
        ordered.extend((pen, points) for points in routed)
        current = routed[-1][-1]
    if _travel_distance(ordered, origin) < _travel_distance(paths, origin):
        return tuple(ordered)
    return paths


def _effective_pens(paths: tuple[PenPath, ...]) -> Iterator[PenPath]:
    active = None
    for pen, points in paths:
        if pen is not None:
            active = pen
        yield active, points


def _nearest_paths(strokes: tuple[Polyline, ...], start: Point) -> list[Polyline]:
    if len(strokes) == 1:
        (stroke,) = strokes
        return [
            stroke[::-1] if dist(start, stroke[-1]) < dist(start, stroke[0]) else stroke
        ]
    endpoints = np.asarray([(stroke[0], stroke[-1]) for stroke in strokes], dtype=float)
    remaining = np.arange(len(strokes))
    ordered = []
    current = start
    while remaining.size:
        delta = endpoints[remaining] - current
        distance = np.hypot(delta[..., 0], delta[..., 1])
        index, reverse = divmod(int(np.argmin(distance)), 2)
        stroke = strokes[remaining[index]]
        points = stroke[::-1] if reverse else stroke
        ordered.append(points)
        current = points[-1]
        remaining = np.delete(remaining, index)
    return ordered


def _travel_distance(
    paths: list[PenPath] | tuple[PenPath, ...], origin: Point
) -> float:
    if not paths:
        return 0.0
    return dist(origin, paths[0][1][0]) + sum(
        dist(first[-1], second[0]) for (_, first), (_, second) in pairwise(paths)
    )
