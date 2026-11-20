"""Expression tree utilities for resolving render-time parameters."""

from __future__ import annotations

__docformat__ = "google"

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any


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
        return Expr("maximum", (self, other))

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
    return Expr("maximum", (left, right))


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


def evaluate_expr(expr, ctx, params, *, track_grad=False):
    """Resolve a reusable expression against this execution's parameters.

    Learning executions preserve Tensor leaves; ordinary rendering snapshots
    them as Python values suitable for shader uniforms. No tape is cached on
    the expression itself.
    """
    from ..learning.autodiff import Parameter, Tensor

    if isinstance(expr, Parameter) and expr.animate is not None:
        progress = getattr(ctx, "progress", None)
        if progress is None:
            progress = getattr(ctx, "inputs", {}).get("progress", 0.0)
        resolve = getattr(ctx, "resolve_animated", None)
        value = resolve(expr, progress) if callable(resolve) else expr.at(progress)
        if track_grad:
            return value
        array = value.numpy()
        return array.item() if array.ndim == 0 else array.tolist()

    if isinstance(expr, Tensor):
        if track_grad:
            return expr
        value = expr.numpy()
        return value.item() if value.ndim == 0 else value.tolist()

    if isinstance(expr, str):
        parsed = _parse_context_percent(expr)
        if parsed is not None:
            return evaluate_expr(parsed, ctx, params, track_grad=track_grad)
        return expr

    if not isinstance(expr, Expr):
        return expr

    op = expr.op
    args = expr.args
    if op in {"param", "input"}:
        return _resolve_binding(op, args, ctx, params, track_grad)

    values = tuple(
        evaluate_expr(arg, ctx, params, track_grad=track_grad) for arg in args
    )
    return _apply_expression_op(op, values, ctx)


def _apply_expression_op(op, values, ctx):
    from ..learning.autodiff import Tensor

    if op in {"percent", "percent_x", "percent_y", "percent_of_min"}:
        return _evaluate_percent(op, values[0], ctx)
    return _apply_math_op(op, values, Tensor)


def _resolve_binding(op, args, ctx, params, track_grad):
    if op == "param":
        name, default = args
        if name in params:
            return evaluate_expr(params[name], ctx, params, track_grad=track_grad)
        if default is not None:
            return evaluate_expr(default, ctx, params, track_grad=track_grad)
        raise KeyError(f"Missing render parameter '{name}'")
    name = args[0]
    inputs = getattr(ctx, "inputs", {})
    if name not in inputs:
        raise KeyError(f"Missing render input '{name}'")
    return inputs[name]


def _evaluate_percent(op, value, ctx):
    return {
        "percent": ctx.percent,
        "percent_x": ctx.percent_x,
        "percent_y": ctx.percent_y,
        "percent_of_min": ctx.percent_of_min,
    }[op](value)


def _apply_math_op(op, values, tensor_type):
    if op == "select_positive":
        test, positive, otherwise = values
        if any(isinstance(value, tensor_type) for value in values):
            from ..learning.autodiff import where

            condition = test.data if isinstance(test, tensor_type) else test
            return where(condition > 0.0, positive, otherwise)
        return _select_positive(test, positive, otherwise)

    if op == "index":
        value, index = values
        if isinstance(value, tensor_type):
            return value[index]
        for part in index if isinstance(index, tuple) else (index,):
            value = value[part]
        return value

    if op in {"sin", "cos"}:
        return _evaluate_trig(op, values[0], tensor_type)
    if op in {"neg", "pow", "abs"}:
        return _evaluate_simple_math(op, values)
    if op in {"relu", "softplus", "sigmoid", "tanh", "sqrt"}:
        return _evaluate_activation(op, values[0], tensor_type)
    if op in {"maximum", "minimum"}:
        return _evaluate_minmax(op, values, tensor_type)
    binary = {
        "add": lambda left, right: left + right,
        "sub": lambda left, right: left - right,
        "mul": lambda left, right: left * right,
        "div": lambda left, right: left / right,
    }
    if op in binary:
        return binary[op](*values)

    raise ValueError(f"Unknown expression op '{op}'")


def _evaluate_trig(op, value, tensor_type):
    if isinstance(value, tensor_type):
        return getattr(value, op)()
    return getattr(math, op)(value)


def _evaluate_simple_math(op, values):
    if op == "neg":
        return -values[0]
    if op == "pow":
        return values[0] ** values[1]
    return abs(values[0])


def _evaluate_activation(op, value, tensor_type):
    if isinstance(value, tensor_type):
        return getattr(value, op)()
    if op == "relu":
        return max(value, 0.0)
    if op == "softplus":
        return max(value, 0.0) + math.log1p(math.exp(-abs(value)))
    if op == "sigmoid":
        return _scalar_sigmoid(value)
    if op == "tanh":
        return math.tanh(value)
    return math.sqrt(value)


def _scalar_sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _evaluate_minmax(op, values, tensor_type):
    if any(isinstance(value, tensor_type) for value in values):
        from ..learning.autodiff import maximum, minimum

        return (maximum if op == "maximum" else minimum)(*values)
    return (max if op == "maximum" else min)(values)


def _parse_context_percent(value):
    if not value.endswith(("%", "%x", "%y", "%min")):
        return None

    if value.endswith("%min"):
        return percent_of_min(float(value[:-4]))
    if value.endswith("%x"):
        return percent_x(float(value[:-2]))
    if value.endswith("%y"):
        return percent_y(float(value[:-2]))
    if value.endswith("%"):
        return percent(float(value[:-1]))

    return None
