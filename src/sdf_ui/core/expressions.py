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


def param(name: str, default=None):
    return Expr("param", (name, default))


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


def evaluate_expr(expr, ctx, params):
    if isinstance(expr, str):
        parsed = _parse_context_percent(expr)
        if parsed is not None:
            return evaluate_expr(parsed, ctx, params)
        return expr

    if not isinstance(expr, Expr):
        return expr

    op = expr.op
    args = expr.args

    if op == "param":
        name, default = args
        if name in params:
            return params[name]
        if default is not None:
            return default
        raise KeyError(f"Missing render parameter '{name}'")

    values = tuple(evaluate_expr(arg, ctx, params) for arg in args)

    if op == "percent":
        return ctx.percent(values[0])
    if op == "percent_x":
        return ctx.percent_x(values[0])
    if op == "percent_y":
        return ctx.percent_y(values[0])
    if op == "percent_of_min":
        return ctx.percent_of_min(values[0])
    if op == "sin":
        return math.sin(values[0])
    if op == "cos":
        return math.cos(values[0])
    if op == "neg":
        return -values[0]
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
