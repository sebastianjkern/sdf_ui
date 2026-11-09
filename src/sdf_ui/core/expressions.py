"""Expression tree utilities for resolving render-time parameters."""

__docformat__ = "google"

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple

    def __add__(self, other):
        return Expr("add", (self, other))

    def __radd__(self, other):
        return Expr("add", (other, self))

    def __sub__(self, other):
        return Expr("sub", (self, other))

    def __rsub__(self, other):
        return Expr("sub", (other, self))

    def __mul__(self, other):
        return Expr("mul", (self, other))

    def __rmul__(self, other):
        return Expr("mul", (other, self))

    def __truediv__(self, other):
        return Expr("div", (self, other))

    def __rtruediv__(self, other):
        return Expr("div", (other, self))

    def __neg__(self):
        return Expr("neg", (self,))

    def __pow__(self, exponent):
        return Expr("pow", (self, exponent))

    def __abs__(self):
        return Expr("abs", (self,))

    def relu(self):
        return Expr("relu", (self,))

    def softplus(self):
        return Expr("softplus", (self,))

    def sigmoid(self):
        return Expr("sigmoid", (self,))

    def tanh(self):
        return Expr("tanh", (self,))

    def sqrt(self):
        return Expr("sqrt", (self,))

    def maximum(self, other):
        return Expr("maximum", (self, other))

    def minimum(self, other):
        return Expr("minimum", (self, other))


def param(name: str, default=None):
    return Expr("param", (name, default))


def input_value(name: str):
    """A non-trainable scalar input supplied to a compiled graph."""
    if not isinstance(name, str) or not name:
        raise ValueError("Input name must be a nonempty string")
    return Expr("input", (name,))


def percent(value):
    return Expr("percent", (value,))


def percent_x(value):
    return Expr("percent_x", (value,))


def percent_y(value):
    return Expr("percent_y", (value,))


def percent_of_min(value):
    return Expr("percent_of_min", (value,))


def sin(value):
    return Expr("sin", (value,))


def cos(value):
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

    if op == "param":
        name, default = args
        if name in params:
            return evaluate_expr(params[name], ctx, params, track_grad=track_grad)
        if default is not None:
            return evaluate_expr(default, ctx, params, track_grad=track_grad)
        raise KeyError(f"Missing render parameter '{name}'")

    if op == "input":
        name = args[0]
        inputs = getattr(ctx, "inputs", {})
        if name not in inputs:
            raise KeyError(f"Missing render input '{name}'")
        return inputs[name]

    values = tuple(
        evaluate_expr(arg, ctx, params, track_grad=track_grad) for arg in args
    )

    if op == "index":
        value, index = values
        if isinstance(value, Tensor):
            return value[index]
        for part in index if isinstance(index, tuple) else (index,):
            value = value[part]
        return value

    if op == "percent":
        return ctx.percent(values[0])
    if op == "percent_x":
        return ctx.percent_x(values[0])
    if op == "percent_y":
        return ctx.percent_y(values[0])
    if op == "percent_of_min":
        return ctx.percent_of_min(values[0])
    if op == "sin":
        return values[0].sin() if isinstance(values[0], Tensor) else math.sin(values[0])
    if op == "cos":
        return values[0].cos() if isinstance(values[0], Tensor) else math.cos(values[0])
    if op == "neg":
        return -values[0]
    if op == "pow":
        return values[0] ** values[1]
    if op == "abs":
        return abs(values[0])
    if op in {"relu", "softplus", "sigmoid", "tanh", "sqrt"}:
        value = values[0]
        if isinstance(value, Tensor):
            return getattr(value, op)()
        if op == "relu":
            return max(value, 0.0)
        if op == "softplus":
            return max(value, 0.0) + math.log1p(math.exp(-abs(value)))
        if op == "sigmoid":
            if value >= 0:
                return 1.0 / (1.0 + math.exp(-value))
            exponential = math.exp(value)
            return exponential / (1.0 + exponential)
        if op == "tanh":
            return math.tanh(value)
        return math.sqrt(value)
    if op in {"maximum", "minimum"}:
        if any(isinstance(value, Tensor) for value in values):
            from ..learning.autodiff import maximum, minimum

            return (maximum if op == "maximum" else minimum)(*values)
        return (max if op == "maximum" else min)(values)
    if op == "add":
        return values[0] + values[1]
    if op == "sub":
        return values[0] - values[1]
    if op == "mul":
        return values[0] * values[1]
    if op == "div":
        return values[0] / values[1]

    raise ValueError(f"Unknown expression op '{op}'")


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
