"""Expression tree node and constructors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

_MAXIMUM_TEXT = 'maximum'


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple[Any, ...]

    def __add__(self, other: Any) -> Expr:
        return Expr("add", (self, other))

    def __radd__(self, other: Any) -> Expr:
        return Expr("add", (other, self))

    def __sub__(self, other: Any) -> Expr:
        return Expr("sub", (self, other))

    def __rsub__(self, other: Any) -> Expr:
        return Expr("sub", (other, self))

    def __mul__(self, other: Any) -> Expr:
        return Expr("mul", (self, other))

    def __rmul__(self, other: Any) -> Expr:
        return Expr("mul", (other, self))

    def __truediv__(self, other: Any) -> Expr:
        return Expr("div", (self, other))

    def __rtruediv__(self, other: Any) -> Expr:
        return Expr("div", (other, self))

    def __neg__(self) -> Expr:
        return Expr("neg", (self,))

    def __pow__(self, exponent: Any) -> Expr:
        return Expr("pow", (self, exponent))

    def __abs__(self) -> Expr:
        return Expr("abs", (self,))

    def relu(self) -> Expr:
        return Expr("relu", (self,))

    def softplus(self) -> Expr:
        return Expr("softplus", (self,))

    def sigmoid(self) -> Expr:
        return Expr("sigmoid", (self,))

    def tanh(self) -> Expr:
        return Expr("tanh", (self,))

    def sqrt(self) -> Expr:
        return Expr("sqrt", (self,))

    def maximum(self, other: Any) -> Expr:
        return Expr(_MAXIMUM_TEXT, (self, other))

    def minimum(self, other: Any) -> Expr:
        return Expr("minimum", (self, other))

    def clamp(self, low: Any = 0.0, high: Any = 1.0) -> Expr:
        """Keep this expression inside an interval."""
        return self.maximum(low).minimum(high)

    def remap(self, source, target, *, clamp: bool = True) -> Expr:
        """Map the source interval to the target interval."""
        lo, hi = source
        out_lo, out_hi = target
        position = (self - lo) / (hi - lo)
        if clamp:
            position = position.clamp()
        return out_lo + position * (out_hi - out_lo)


def _lazy_scalar(value):
    """Keep literal arithmetic cheap and defer arithmetic on live leaves."""
    if isinstance(value, (Real, Expr)):
        return value
    return Expr("add", (value, 0.0))


def _scalar_minimum(left, right):
    if isinstance(left, Real) and isinstance(right, Real):
        return min(left, right)
    return Expr("minimum", (left, right))


def _scalar_maximum(left, right):
    if isinstance(left, Real) and isinstance(right, Real):
        return max(left, right)
    return Expr(_MAXIMUM_TEXT, (left, right))


def _select_positive(test, positive, otherwise):
    if isinstance(test, Real):
        return positive if test > 0.0 else otherwise
    return Expr("select_positive", (test, positive, otherwise))


def param(name: str, default: Any = None) -> Expr:
    """Reference a named render parameter, with an optional fallback."""
    if not isinstance(name, str) or not name:
        raise ValueError("Parameter name must be a nonempty string")
    return Expr("param", (name, default))


def input_value(name: str) -> Expr:
    """A non-trainable scalar input supplied to a compiled graph."""
    if not isinstance(name, str) or not name:
        raise ValueError("Input name must be a nonempty string")
    return Expr("input", (name,))


def validate_progress(progress: Any) -> float:
    """Validate normalized animation progress."""
    if isinstance(progress, bool) or not isinstance(progress, Real):
        raise ValueError("progress must be a finite number in [0, 1]")
    progress = float(progress)
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be a finite number in [0, 1]")
    return progress


def percent(value: Any) -> Expr:
    """Scale ``value`` by the canvas width divided by 100."""
    return Expr("percent", (value,))


def percent_x(value: Any) -> Expr:
    """Scale ``value`` by the canvas width divided by 100."""
    return Expr("percent_x", (value,))


def percent_y(value: Any) -> Expr:
    """Scale ``value`` by the canvas height divided by 100."""
    return Expr("percent_y", (value,))


def percent_of_min(value: Any) -> Expr:
    """Scale ``value`` by the smaller canvas dimension divided by 100."""
    return Expr("percent_of_min", (value,))


def sin(value: Any) -> Expr:
    return Expr("sin", (value,))


def cos(value: Any) -> Expr:
    return Expr("cos", (value,))


def ease_in_cubic(value: Any) -> Expr:
    """Build a cubic ease-in expression for normalized progress."""
    return Expr("pow", (value, 3.0))


def ease_out_cubic(value: Any) -> Expr:
    """Build a cubic ease-out expression for normalized progress."""
    inverse = Expr("sub", (1.0, value))
    return Expr("sub", (1.0, Expr("pow", (inverse, 3.0))))


def ease_in_out_sine(value: Any) -> Expr:
    """Build a sine ease-in-out expression for normalized progress."""
    return (1.0 - cos(math.pi * value)) / 2.0
