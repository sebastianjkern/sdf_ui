"""Shared positional scales for the chart grammar."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Any

from ._plot_sampling import Point, _ordered_range
from .core.expressions import Expr


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
        from .learning.autodiff import Parameter

        values = tuple(values)
        categories = tuple(
            dict.fromkeys(value for value in values if isinstance(value, str))
        )
        if categories:
            if not all(isinstance(value, str) for value in values):
                raise ValueError(f"{axis} cannot mix categorical and numeric values")
            if limits is not None:
                raise ValueError(f"numeric {axis} limits do not apply to categories")
            if include_zero:
                raise ValueError(f"{axis} must be numeric for column heights")
            return cls((-0.5, len(categories) - 0.5), categories)

        numbers = []
        for value in values:
            if isinstance(value, Real) and not isinstance(value, bool):
                if not math.isfinite(value):
                    raise ValueError(f"{axis} data must be finite")
                numbers.append(float(value))
            elif isinstance(value, (Expr, Parameter)):
                if limits is None:
                    raise ValueError(
                        f"live {axis} data requires explicit {axis} limits"
                    )
            else:
                raise TypeError(
                    f"{axis} data must contain numbers, expressions, or strings"
                )
        if limits is not None:
            return cls(_ordered_range(limits, f"{axis} limits"))
        if include_zero:
            numbers.append(0.0)
        if not numbers:
            raise ValueError(f"chart has no {axis} data")
        low, high = min(numbers), max(numbers)
        if low == high:
            margin = max(0.5, abs(low) * 0.1)
            low, high = low - margin, high + margin
        else:
            margin = (high - low) * padding
            low = low if include_zero and low == 0 else low - margin
            high = high if include_zero and high == 0 else high + margin
        return cls(_ordered_range((low, high), f"{axis} limits"))

    def encode(self, value: Any) -> Any:
        return self.categories.index(value) if self.categories else value

    def ticks(self) -> range | None:
        return range(len(self.categories)) if self.categories else None

    def label(self, value: float) -> str:
        return self.categories[int(value)] if self.categories else f"{value:g}"
