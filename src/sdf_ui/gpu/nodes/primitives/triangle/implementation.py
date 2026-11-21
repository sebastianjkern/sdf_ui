"""Triangle shader's uniform layout and active-edge differentiation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)

from ..segment_math import minimum_candidates, segment


def reference(context, inputs, params):
    import numpy as np

    from sdf_ui.learning.render import _tensor

    points = [_tensor(params[name]) for name in ("point0", "point1", "point2")]
    e0, e2 = points[1] - points[0], points[0] - points[2]
    orientation = np.sign((e0[0] * e2[1] - e0[1] * e2[0]).data)
    if orientation == 0:
        raise ValueError("triangle requires non-collinear vertices for differentiation")
    distances, sides = [], []
    for a, b in zip(points, points[1:] + points[:1]):
        distances.append(segment(context.x, context.y, a[0], a[1], b[0], b[1]))
        x, y = context.x - a[0], context.y - a[1]
        edge = b - a
        sides.append(orientation * (x * edge[1] - y * edge[0]).data)
    return -minimum_candidates(*distances) * np.sign(np.minimum.reduce(sides))


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    uniforms=tuple(Uniform(name, "vec2") for name in ("point0", "point1", "point2")),
)
CONTRACT = ShaderDifferentiation(
    "triangle", "gpu/nodes/primitives/triangle/triangle.wgsl", (),
    tuple(ShaderArgument(name, (2,), "vec2") for name in ("point0", "point1", "point2")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Active edges/signs fixed; degenerate triangles outside shader domain.", reference=reference,
)
