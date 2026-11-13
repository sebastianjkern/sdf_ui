"""Expression tree utilities for resolving render-time parameters."""

from __future__ import annotations

__docformat__ = "google"

import math
from dataclasses import dataclass
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


def evaluate_expr(expr, ctx, params, *, track_grad=False):
    """Resolve a reusable expression against this execution's parameters.

    Learning executions preserve Tensor leaves; ordinary rendering snapshots
    them as Python values suitable for shader uniforms. No tape is cached on
    the expression itself.
    """
    from ..learning.autodiff import Tensor

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
