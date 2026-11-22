"""Sample pen trajectories and clip them to an SDF or a shaded region."""

import math
from dataclasses import dataclass

import numpy as np

from .vector import HATCH_LINE_LIMIT, Point, Polyline

_SAMPLE_STEP = 0.5
_VERTEX_STEP = 4.0
_MIN_LENGTH = 0.05
_MAX_SAMPLES = 20_000_000
_MAX_JITTER_FACTOR = 1.5


@dataclass(frozen=True)
class StrokeDrift:
    bow: float
    jitter: float
    seed: int


@dataclass(frozen=True)
class HatchFrame:
    angle: float
    spacing: float
    phase: float
    origin: Point


def relief_shadow(distance, level, direction, depth):
    """Shadow strength from an SDF shaped into a rounded relief."""
    inward = np.clip((level - distance) / depth, 0, 1)
    height = depth * np.sqrt(inward * (2 - inward))
    sy = np.gradient(height, axis=0) if height.shape[0] > 1 else np.zeros_like(height)
    sx = np.gradient(height, axis=1) if height.shape[1] > 1 else np.zeros_like(height)
    light = np.asarray(direction, dtype=float)
    light /= math.hypot(*direction)
    illumination = (light[2] - sx * light[0] - sy * light[1]) / np.sqrt(
        1 + sx * sx + sy * sy
    )
    return 1 - np.clip(illumination, 0, 1)


def _bilinear(field, points):
    height, width = field.shape
    x, y = points.T
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = x - x0, y - y0
    top = field[y0, x0] * (1 - fx) + field[y0, x1] * fx
    bottom = field[y1, x0] * (1 - fx) + field[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def _clip_field(field, points):
    height, width = field.shape
    x, y = points.T
    border = np.maximum.reduce(
        (-0.5 - x, x - (width - 0.5), -0.5 - y, y - (height - 0.5))
    )
    return np.maximum(border, _bilinear(field, points))


def _crossing(field, start, end, inside):
    for _ in range(10):
        middle = (start + end) / 2
        if bool(_clip_field(field, middle[None])[0] <= 0) == inside:
            start = middle
        else:
            end = middle
    return (start + end) / 2


def _stroke_runs(field, points, curved):
    active = _clip_field(field, points) <= 0
    edges = np.flatnonzero(np.diff(np.r_[False, active, False]))
    stride = max(1, round(_VERTEX_STEP / _SAMPLE_STEP))
    for start, stop in edges.reshape(-1, 2):
        first = (
            _crossing(field, points[start - 1], points[start], False)
            if start
            else points[start]
        )
        last = (
            _crossing(field, points[stop - 1], points[stop], True)
            if stop < len(points)
            else points[-1]
        )
        if stop - start <= 1 and np.linalg.norm(last - first) < _MIN_LENGTH:
            continue
        interior = (
            points[start + stride // 2 : stop - 1 : stride] if curved else points[:0]
        )
        chain = np.vstack((first, interior, last))
        yield tuple((float(x), float(y)) for x, y in chain)


def _stroke_offset(index, count, drift):
    if drift is None or (drift.bow == 0 and drift.jitter == 0):
        return 0.0
    random = np.random.default_rng(
        np.random.SeedSequence([drift.seed, index % (2**32)])
    )
    shift, tilt, bend = random.uniform(-1, 1, 3)
    u = np.linspace(-1, 1, count)
    return drift.jitter * (shift + tilt * u / 2) + drift.bow * bend * (1 - u * u)


def _stroke_domain(shape, frame, drift, direction, normal):
    height, width = shape
    corners = np.asarray(
        (
            (-0.5, -0.5),
            (width - 0.5, -0.5),
            (width - 0.5, height - 0.5),
            (-0.5, height - 0.5),
        )
    )
    relative = corners - frame.origin
    along, across = relative @ direction, relative @ normal
    reach = drift.bow + _MAX_JITTER_FACTOR * drift.jitter if drift else 0.0
    low = math.ceil((across.min() - reach - frame.phase) / frame.spacing)
    high = math.floor((across.max() + reach - frame.phase) / frame.spacing)
    count = math.ceil((along.max() - along.min()) / _SAMPLE_STEP) + 1
    lines = max(0, high - low + 1)
    if lines > HATCH_LINE_LIMIT or count * max(lines, 1) > _MAX_SAMPLES:
        raise ValueError("Hatch resolution exceeds the stroke sampling limit")
    return range(low, high + 1), np.linspace(along.min(), along.max(), count)


def stroke_lines(
    field, frame: HatchFrame, drift: StrokeDrift | None
) -> tuple[Polyline, ...]:
    """Draw gently bowed strokes and split them at viewport and mask crossings."""
    direction = np.array((math.cos(frame.angle), math.sin(frame.angle)))
    normal = np.array((-direction[1], direction[0]))
    indices, along = _stroke_domain(field.shape, frame, drift, direction, normal)
    base = np.asarray(frame.origin) + along[:, None] * direction
    lines = []
    for index in indices:
        offset = (
            index * frame.spacing
            + frame.phase
            + _stroke_offset(index, len(along), drift)
        )
        lines.extend(
            _stroke_runs(
                field,
                base + np.asarray(offset)[..., None] * normal,
                curved=drift is not None and drift.bow > 0,
            )
        )
    return tuple(lines)
