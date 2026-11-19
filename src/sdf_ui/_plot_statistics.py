"""Statistics calculated before fitting plot positions."""

import math
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from ._plot_sampling import _ordered_range


@dataclass(frozen=True)
class Histogram:
    bins: int

    def __post_init__(self):
        if (
            isinstance(self.bins, bool)
            or not isinstance(self.bins, Integral)
            or self.bins < 1
        ):
            raise ValueError("histogram bins must be a positive integer")

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> tuple[float, ...]:
        values = []
        for row in rows:
            value = row["x"]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError("histograms require finite numeric observations")
            values.append(float(value))
        if not values:
            return ()
        low, high = min(values), max(values)
        if low == high:
            margin = max(0.5, abs(low) * 0.05)
            low, high = low - margin, high + margin
        low, high = _ordered_range((low, high), "histogram domain")
        edges = tuple(
            low + (high - low) * (i / self.bins) for i in range(self.bins)
        ) + (high,)
        if any(a >= b for a, b in zip(edges, edges[1:])):
            raise ValueError("histogram bins are too narrow to represent")
        return edges

    @staticmethod
    def apply(rows, edges):
        counts = [0] * (len(edges) - 1)
        for row in rows:
            index = min(bisect_right(edges, row["x"]) - 1, len(counts) - 1)
            counts[index] += 1
        return [
            {"xmin": left, "xmax": right, "ymin": 0.0, "ymax": count}
            for left, right, count in zip(edges, edges[1:], counts)
            if count
        ]
