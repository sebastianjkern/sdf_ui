"""Shared positional scales for the chart grammar."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Any

from ._plot_sampling import Point, _ordered_range
from .core.expressions import Expr

_CONSTANT_DATA_MARGIN = 0.5
_RELATIVE_DATA_MARGIN = 0.1


def _categorical_scale(values, limits, axis, include_zero):
    categories = tuple(dict.fromkeys(value for value in values if isinstance(value, str)))
    if not categories:
        return None
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"{axis} cannot mix categorical and numeric values")
    if limits is not None:
        raise ValueError(f"numeric {axis} limits do not apply to categories")
    if include_zero:
        raise ValueError(f"{axis} must be numeric for column heights")
    return categories


def _numeric_values(values, limits, axis):
    from .learning.autodiff import Parameter

    numbers = []
    for value in values:
        if isinstance(value, Real) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ValueError(f"{axis} data must be finite")
            numbers.append(float(value))
        elif isinstance(value, (Expr, Parameter)):
            if limits is None:
                raise ValueError(f"live {axis} data requires explicit {axis} limits")
        else:
            raise TypeError(f"{axis} data must contain numbers, expressions, or strings")
    return numbers


def _padded_limits(numbers, axis, include_zero, padding):
    if include_zero:
        numbers.append(0.0)
    if not numbers:
        raise ValueError(f"chart has no {axis} data")
    low, high = min(numbers), max(numbers)
    if low == high:
        margin = max(_CONSTANT_DATA_MARGIN, abs(low) * _RELATIVE_DATA_MARGIN)
        low, high = low - margin, high + margin
    else:
        margin = (high - low) * padding
        low = low if include_zero and low == 0 else low - margin
        high = high if include_zero and high == 0 else high + margin
    return _ordered_range((low, high), f"{axis} limits")


@dataclass(frozen=True)
class PositionScale:
    limits: Point
    categories: tuple[str, ...] = ()

    @classmethod
    def fit(
        cls,
        values: Iterable[Any],
        limits: Point | None,
        *,
        axis: str,
        include_zero: bool = False,
        padding: float = 0.06,
    ) -> "PositionScale":
        values = tuple(values)
        categories = _categorical_scale(values, limits, axis, include_zero)
        if categories:
            return cls((-0.5, len(categories) - 0.5), categories)
        numbers = _numeric_values(values, limits, axis)
        if limits is not None:
            return cls(_ordered_range(limits, f"{axis} limits"))
        return cls(_padded_limits(numbers, axis, include_zero, padding))

    def encode(self, value: Any) -> Any:
        return self.categories.index(value) if self.categories else value

    def ticks(self) -> range | None:
        return range(len(self.categories)) if self.categories else None

    def label(self, value: float) -> str:
        return self.categories[int(value)] if self.categories else f"{value:g}"
