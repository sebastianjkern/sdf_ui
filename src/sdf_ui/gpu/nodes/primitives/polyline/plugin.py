"""Evaluate distances to line segments using a bounding-box hierarchy."""

import math
import struct

import wgpu

from sdf_ui.core.sdf import SDFTexture
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind

_HEADER = struct.Struct("<fI")
_NODE = struct.Struct("<8f2I")
# Keep squared distances representable in the GPU's float32 arithmetic.
_COORDINATE_LIMIT = 1e15


def _path_buffer(points, radius):
    segments = []
    previous = None
    for point in points:
        if point is None:
            previous = None
            continue
        x, y = map(float, point)
        if not all(math.isfinite(v) and abs(v) <= _COORDINATE_LIMIT for v in (x, y)):
            raise ValueError("polyline points must contain finite canvas coordinates")
        current = (x, y)
        if previous is not None:
            segments.append((previous, current))
        previous = current

    count = max(0, len(segments) * 2 - 1)
    payload = bytearray(_HEADER.size + max(1, count) * _NODE.size)
    _HEADER.pack_into(payload, 0, radius, count)
    next_index = 0

    def visit(items):
        nonlocal next_index
        index = next_index
        next_index += 1
        low_x = min(min(a[0], b[0]) for a, b in items)
        low_y = min(min(a[1], b[1]) for a, b in items)
        high_x = max(max(a[0], b[0]) for a, b in items)
        high_y = max(max(a[1], b[1]) for a, b in items)
        leaf = len(items) == 1
        a, b = items[0]
        if not leaf:
            axis = int(high_y - low_y > high_x - low_x)
            items.sort(key=lambda segment: segment[0][axis] + segment[1][axis])
            middle = len(items) // 2
            visit(items[:middle])
            visit(items[middle:])
        # The next index skips this entire subtree when its box is too far.
        _NODE.pack_into(
            payload,
            _HEADER.size + index * _NODE.size,
            low_x,
            low_y,
            high_x,
            high_y,
            *a,
            *b,
            next_index,
            leaf,
        )

    if segments:
        visit(segments)
    return payload


def render_polyline(renderer, _inputs, params):
    ctx = renderer.ctx
    radius = float(params["radius"])
    if not math.isfinite(radius) or not 0 <= radius <= _COORDINATE_LIMIT:
        raise ValueError("polyline radius must be finite and nonnegative")
    payload = _path_buffer(params["points"], radius)
    buffer = ctx.create_buffer(
        payload, usage=wgpu.BufferUsage.STORAGE, label="polyline"
    )
    try:
        output = ctx.r32f()
        try:
            ctx.run_shader("polyline", {0: output, 1: buffer})
        except BaseException:
            output.close()
            raise
    finally:
        if not ctx._defer_resource_destroy(buffer):
            buffer.destroy()
    return SDFTexture(tex=output, context=ctx)


def register_plugins(registry):
    registry.register(
        Plugin(
            "polyline",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("points", "radius"),
            defaults={"radius": 0.0},
            render_func=render_polyline,
            public=True,
        )
    )
