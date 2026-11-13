"""Evaluation rectangles for cropped, pointwise GPU renders."""

from __future__ import annotations

import math
import re
import struct
from numbers import Integral


def normalize_rect(rect, canvas_size):
    """Validate an ``(x, y, width, height)`` rectangle in canvas pixels."""
    if rect is None:
        return None
    try:
        values = tuple(rect)
    except TypeError as error:
        raise TypeError("evaluation_rect must be (x, y, width, height)") from error
    if len(values) != 4 or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise TypeError("evaluation_rect must contain four integers")
    x, y, width, height = map(int, values)
    canvas_width, canvas_height = canvas_size
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > canvas_width
        or y + height > canvas_height
    ):
        raise ValueError(
            "evaluation_rect must be a positive rectangle inside the canvas"
        )
    return x, y, width, height


def rect_groups(code, rect):
    """Calculate dispatch groups from a shader's declared workgroup size."""
    match = re.search(r"@workgroup_size\((\d+)\s*,\s*(\d+)", code)
    if match is None:
        raise ValueError("Region evaluation requires a two-dimensional compute shader")
    local_width, local_height = map(int, match.groups())
    _, _, width, height = rect
    return math.ceil(width / local_width), math.ceil(height / local_height), 1


def pack_rect(canvas_size, rect, inputs):
    """Pack canvas dimensions, output region, and input image origins."""
    x, y, width, height = rect
    values = (*canvas_size, x, y, width, height)
    for image in inputs:
        values += tuple(getattr(image, "origin", (0, 0)))
    return struct.pack("<" + "I" * len(values), *values)


def specialize_wgsl_for_rect(code, interface, binding):
    """Make a full-canvas WGSL kernel address a cropped output and inputs."""
    input_origins = {
        item.name: f"inputOrigin{index}" for index, item in enumerate(interface.inputs)
    }
    outputs = (interface.output, *interface.extra_outputs)
    for output in outputs:
        code = _rewrite_texture_calls(code, "textureStore", output.name, "outputOrigin")
    for name, origin in input_origins.items():
        code = _rewrite_texture_calls(code, "textureLoad", name, origin)
    code = re.sub(
        r"textureDimensions\(\s*\w+\s*\)",
        "_sdfRegion.canvasSize",
        code,
    )
    code = _offset_entry_point(code)

    fields = [
        "canvasSize: vec2<u32>,",
        "outputOrigin: vec2<u32>,",
        "outputExtent: vec2<u32>,",
    ]
    fields.extend(f"{name}: vec2<u32>," for name in input_origins.values())
    region_struct = (
        "struct SdfEvaluationRegion {\n    "
        + "\n    ".join(fields)
        + "\n};\n"
        + f"@group(0) @binding({binding}) var<uniform> _sdfRegion: SdfEvaluationRegion;\n"
    )
    return region_struct + code


def _rewrite_texture_calls(code, function, image_name, origin_name):
    token = function + "("
    search_from = 0
    while True:
        start = code.find(token, search_from)
        if start < 0:
            return code
        open_paren = start + len(function)
        close_paren = _matching_paren(code, open_paren)
        args = _split_arguments(code[open_paren + 1 : close_paren])
        if args and args[0].strip() == image_name and len(args) >= 2:
            coordinate = args[1].strip()
            args[1] = f"({coordinate} - vec2<i32>(_sdfRegion.{origin_name}))"
            replacement = token + ", ".join(args) + ")"
            code = code[:start] + replacement + code[close_paren + 1 :]
            search_from = start + len(replacement)
        else:
            search_from = close_paren + 1


def _matching_paren(code, open_paren):
    depth = 0
    for index in range(open_paren, len(code)):
        if code[index] == "(":
            depth += 1
        elif code[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Malformed WGSL function call")


def _split_arguments(source):
    args = []
    start = 0
    depth = 0
    for index, character in enumerate(source):
        if character in "([<":
            depth += 1
        elif character in ")]>" and depth:
            depth -= 1
        elif character == "," and depth == 0:
            args.append(source[start:index])
            start = index + 1
    args.append(source[start:])
    return args


def _offset_entry_point(code):
    match = re.search(r"\bfn\s+compute_main\s*\([^)]*\)[^{]*\{", code)
    if match is None:
        raise ValueError("Region evaluation requires a compute_main entry point")
    open_brace = match.end() - 1
    close_brace = _matching_brace(code, open_brace)
    body = code[open_brace + 1 : close_brace]
    body = re.sub(
        r"\bglobal_invocation_id\.xy\b",
        "(global_invocation_id.xy + _sdfRegion.outputOrigin)",
        body,
    )
    body = re.sub(
        r"\bglobal_invocation_id\.x\b",
        "(global_invocation_id.x + _sdfRegion.outputOrigin.x)",
        body,
    )
    body = re.sub(
        r"\bglobal_invocation_id\.y\b",
        "(global_invocation_id.y + _sdfRegion.outputOrigin.y)",
        body,
    )
    guard = "\n    if (global_invocation_id.x >= _sdfRegion.outputExtent.x || "
    guard += "global_invocation_id.y >= _sdfRegion.outputExtent.y) { return; }\n"
    body = guard + body
    return code[: open_brace + 1] + body + code[close_brace:]


def _matching_brace(code, open_brace):
    depth = 0
    for index in range(open_brace, len(code)):
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Malformed WGSL function body")
