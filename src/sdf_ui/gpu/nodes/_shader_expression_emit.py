"""GLSL emission helpers for typed shader expressions."""

import math


def _emit_expression(op, args, dual):
    match op:
        case "literal":
            return _emit_literal(args[0], dual)
        case "position":
            return "v2(vec2(pos))" if dual else "vec2(texelPos)"
        case "uniform":
            return ("U_" if dual else "") + args[0]
        case "load":
            return (
                f"ld({args[0]},pos,0)" if dual else f"imageLoad(source{args[0]},pos).r"
            )
        case _:
            values = tuple(arg.emit(dual=dual) for arg in args)
            return _emit_operator(op, values, dual)


def _emit_literal(value, dual):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Shader literals must be finite numbers")
    text = repr(float(value))
    return f"c({text})" if dual else text


def _emit_operator(op, values, dual):
    if op == "length2":
        return ("len2" if dual else "length") + f"({values[0]})"
    if op in {"sub", "sub2"}:
        return (
            f"{op}({values[0]},{values[1]})" if dual else f"({values[0]} - {values[1]})"
        )
    if op in {"add", "mul", "div"}:
        return _emit_binary_operator(op, values, dual)
    if op in {"abs", "min", "max", "mix", "neg"}:
        return _emit_function_operator(op, values, dual)
    raise ValueError(f"Unsupported shader expression operation '{op}'")


def _emit_binary_operator(op, values, dual):
    if dual:
        return ("divD" if op == "div" else op) + f"({values[0]},{values[1]})"
    symbol = {"add": "+", "mul": "*", "div": "/"}[op]
    return f"({values[0]} {symbol} {values[1]})"


def _emit_function_operator(op, values, dual):
    if not dual and op == "neg":
        return f"(-{values[0]})"
    names = {"abs": "absD", "min": "minD", "max": "maxD", "mix": "mixD", "neg": "neg"}
    return (names[op] if dual else op) + "(" + ",".join(values) + ")"
