"""Expression tree utilities for resolving render-time parameters."""

from __future__ import annotations

__docformat__ = "google"

import math

from ._expression_nodes import (
    _MAXIMUM_TEXT,
    Expr,
    _select_positive,
    percent,
    percent_of_min,
    percent_x,
    percent_y,
)


def evaluate_expr(expr, ctx, params, *, track_grad=False):
    """Resolve a reusable expression against this execution's parameters.

    Learning executions preserve Tensor leaves; ordinary rendering snapshots
    them as Python values suitable for shader uniforms. No tape is cached on
    the expression itself.
    """
    resolved, value = _resolve_expression_leaf(expr, ctx, params, track_grad)
    if resolved:
        return value

    op = expr.op
    args = expr.args
    if op in {"param", "input"}:
        return _resolve_binding(op, args, ctx, params, track_grad)

    values = tuple(
        evaluate_expr(arg, ctx, params, track_grad=track_grad) for arg in args
    )
    return _apply_expression_op(op, values, ctx)


def _resolve_expression_leaf(expr, ctx, params, track_grad):
    from ..learning.autodiff import Parameter, Tensor

    if isinstance(expr, Parameter) and expr.animate is not None:
        return True, _resolve_animated_parameter(expr, ctx, track_grad)
    if isinstance(expr, Tensor):
        return True, _snapshot_tensor(expr, track_grad)
    if isinstance(expr, str):
        parsed = _parse_context_percent(expr)
        if parsed is not None:
            return True, evaluate_expr(parsed, ctx, params, track_grad=track_grad)
        return True, expr
    if not isinstance(expr, Expr):
        return True, expr
    return False, None


def _resolve_animated_parameter(parameter, ctx, track_grad):
    progress = getattr(ctx, "progress", None)
    if progress is None:
        progress = getattr(ctx, "inputs", {}).get("progress", 0.0)
    resolve = getattr(ctx, "resolve_animated", None)
    value = resolve(parameter, progress) if callable(resolve) else parameter.at(progress)
    return value if track_grad else _tensor_python_value(value)


def _snapshot_tensor(value, track_grad):
    return value if track_grad else _tensor_python_value(value)


def _tensor_python_value(value):
    array = value.numpy()
    return array.item() if array.ndim == 0 else array.tolist()


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
        return _apply_selection(values, tensor_type)

    if op == "index":
        return _apply_index(values, tensor_type)

    if op in {"sin", "cos"}:
        return _evaluate_trig(op, values[0], tensor_type)
    if op in {"neg", "pow", "abs"}:
        return _evaluate_simple_math(op, values)
    if op in {"relu", "softplus", "sigmoid", "tanh", "sqrt"}:
        return _evaluate_activation(op, values[0], tensor_type)
    if op in {_MAXIMUM_TEXT, "minimum"}:
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


def _apply_selection(values, tensor_type):
    test, positive, otherwise = values
    if not any(isinstance(value, tensor_type) for value in values):
        return _select_positive(test, positive, otherwise)
    from ..learning.autodiff import where

    condition = test.data if isinstance(test, tensor_type) else test
    return where(condition > 0.0, positive, otherwise)


def _apply_index(values, tensor_type):
    value, index = values
    if isinstance(value, tensor_type):
        return value[index]
    parts = index if isinstance(index, tuple) else (index,)
    for part in parts:
        value = value[part]
    return value


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

        return (maximum if op == _MAXIMUM_TEXT else minimum)(*values)
    return (max if op == _MAXIMUM_TEXT else min)(values)


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
