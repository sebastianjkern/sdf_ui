"""Triangle shader's uniform layout and active-edge differentiation."""

from sdf_ui.core.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.core.plugins.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.core.shader_expression import ShaderProgram
from ..segment_math import GLSL, MIN3_GLSL, minimum_candidates, segment


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
    "triangle", "core/plugins/primitives/triangle/shader.glsl", (),
    tuple(ShaderArgument(name, (2,), "vec2") for name in ("point0", "point1", "point2")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Active edges/signs fixed; degenerate triangles outside shader domain.", reference=reference,
)
BODY = GLSL + MIN3_GLSL + r"""
void evaluate(ivec2 pos,out D o[8]) {
    D2 p=v2(vec2(pos)), a=U_point0, b=U_point1, q=U_point2;
    D2 e0=sub2(b,a), e1=sub2(q,b), e2=sub2(a,q);
    D2 v0=sub2(p,a), v1=sub2(p,b), vq=sub2(p,q);
    float orientation=sign(e0.x.v*e2.y.v-e0.y.v*e2.x.v);
    float side=min(orientation*(v0.x.v*e0.y.v-v0.y.v*e0.x.v),
               min(orientation*(v1.x.v*e1.y.v-v1.y.v*e1.x.v),
                   orientation*(vq.x.v*e2.y.v-vq.y.v*e2.x.v)));
    D distance=primitiveMin3(primitiveSegment(p,a,b,0.0),
        primitiveSegment(p,b,q,0.0),primitiveSegment(p,q,a,0.0));
    o[0]=mul(distance,c(-sign(side)));
}
"""
PROGRAM = ShaderProgram("triangle", BODY)
