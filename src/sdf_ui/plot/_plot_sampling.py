"""Adaptive sampling and viewport clipping for static plot paths."""

import math
from functools import cache
from numbers import Integral

Point = tuple[float, float]
_MAX_DEPTH = 10


def _ordered_range(values: Point, name: str) -> Point:
    low, high = map(float, values)
    if not math.isfinite(high - low) or not low < high:
        raise ValueError(f"{name} must contain two increasing finite values")
    return low, high


def _clip_segment(a: Point, b: Point, low: Point, high: Point):
    """Clip a segment to a rectangle using its parameter interval."""
    first, last = 0.0, 1.0
    for start, end, lower, upper in zip(a, b, low, high):
        delta = end - start
        if delta == 0:
            if start < lower or start > upper:
                return None
            continue
        enter, exit_ = sorted(((lower - start) / delta, (upper - start) / delta))
        first, last = max(first, enter), min(last, exit_)
        if first > last:
            return None

    def point_at(t):
        if t == 0.0:
            return a
        if t == 1.0:
            return b
        return tuple(
            min(upper, max(lower, start + t * (end - start)))
            for start, end, lower, upper in zip(a, b, low, high)
        )

    return point_at(first), point_at(last)


def _adaptive_segments(sample, start, a, end, b, tolerance, depth=0):
    mid = start + (end - start) / 2
    times = (start + (mid - start) / 2, mid, mid + (end - mid) / 2)
    probes = tuple(sample(t) for t in times)
    points = (a, *probes, b)
    if all(point is None for point in points):
        return
    if all(point is not None for point in points):
        # Measure interpolation error in canvas units. A perpendicular-only
        # test can accept a near-vertical chord through a finite jump.
        if all(
            math.hypot(
                p[0] - (a[0] + (b[0] - a[0]) * ratio),
                p[1] - (a[1] + (b[1] - a[1]) * ratio),
            )
            <= tolerance
            for p, ratio in zip(probes, (0.25, 0.5, 0.75))
        ):
            yield a, b
            return
    # An unresolved interval may contain a discontinuity. Leave a gap.
    if depth < _MAX_DEPTH and start < mid < end:
        yield from _adaptive_segments(
            sample, start, a, mid, probes[1], tolerance, depth + 1
        )
        yield from _adaptive_segments(
            sample, mid, probes[1], end, b, tolerance, depth + 1
        )


def _path_points(segments):
    points = []
    previous = None
    for a, b in segments:
        if previous != a:
            if previous is not None:
                points.append(None)
            points.append(a)
        points.append(b)
        previous = b
    return tuple(points)


def _sample_path(function, domain, project, low, high, tolerance, samples, breaks):
    """Sample once at construction, keeping the probe cache local to this call."""
    start, end = _ordered_range(domain, "domain")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("plot tolerance must be finite and positive")
    if isinstance(samples, bool) or not isinstance(samples, Integral) or samples < 1:
        raise ValueError("plot samples must be a positive integer")
    breaks = tuple(map(float, breaks))
    if not all(map(math.isfinite, breaks)):
        raise ValueError("plot breaks must be finite")
    cuts = (start, *sorted({t for t in breaks if start < t < end}), end)

    @cache
    def sample(t):
        try:
            value = function(t)
        except (ArithmeticError, ValueError):
            return None
        if value is None:
            return None
        p = project(*map(float, value))
        return p if all(map(math.isfinite, p)) else None

    def clipped_segments():
        for index, (a, b) in enumerate(zip(cuts, cuts[1:])):
            yield from _clipped_interval(
                sample, a, b, index, len(cuts), low, high, tolerance, samples
            )

    return _path_points(clipped_segments())


def _clipped_interval(sample, start, end, index, cut_count, low, high, tolerance, samples):
    # Take one-sided samples at explicitly marked discontinuities.
    lo = start if index == 0 else math.nextafter(start, end)
    hi = end if index == cut_count - 2 else math.nextafter(end, start)
    if lo >= hi:
        return
    previous_t, previous_p = lo, sample(lo)
    for step in range(1, samples + 1):
        current_t = hi if step == samples else lo + (hi - lo) * (step / samples)
        current_p = sample(current_t)
        yield from _clipped_adaptive_segments(
            sample, previous_t, previous_p, current_t, current_p,
            low, high, tolerance,
        )
        previous_t, previous_p = current_t, current_p


def _clipped_adaptive_segments(sample, t0, p0, t1, p1, low, high, tolerance):
    for start, end in _adaptive_segments(sample, t0, p0, t1, p1, tolerance):
        clipped = _clip_segment(start, end, low, high)
        if clipped is not None:
            yield clipped
