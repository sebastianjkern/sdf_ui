"""Sector shader's uniform interface and differentiable implementation."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram
from ..angular import GLSL as ANGULAR_GLSL, angles
from ..segment_math import GLSL as SEGMENT_GLSL, MIN3_GLSL, minimum_candidates, segment


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import where
    from sdf_ui.learning.render import _length, _tensor

    center = _tensor(params["center"])
    x, y = context.x - center[0], context.y - center[1]
    start, end, inside = angles(x, y, params)
    radius = _tensor(params["radius"])
    radial = _length(x, y)
    distance = minimum_candidates(
        (radial - radius).abs(),
        segment(x, y, 0.0, 0.0, radius * start.cos(), radius * start.sin(), 1e-6),
        segment(x, y, 0.0, 0.0, radius * end.cos(), radius * end.sin(), 1e-6),
    )
    return where(inside & (radial.data <= radius.data), -distance, distance)


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    uniforms=(Uniform("center", "vec2"), Uniform("radius", "float"),
              Uniform("start_angle", "float"), Uniform("end_angle", "float")),
)
CONTRACT = ShaderDifferentiation(
    "sector", "backends/opengl/nodes/primitives/sector/shader.glsl", (),
    (ShaderArgument("center", (2,), "vec2"), ShaderArgument("radius", (), "float"),
     ShaderArgument("start_angle", (), "float"), ShaderArgument("end_angle", (), "float")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
BODY = ANGULAR_GLSL + SEGMENT_GLSL + MIN3_GLSL + r"""
void evaluate(ivec2 pos,out D o[8]) {
    D2 p=sub2(v2(vec2(pos)),U_center), zero=v2(vec2(0.0));
    D radial=len2(p);
    D distance=primitiveMin3(absD(sub(radial,U_radius)),
        primitiveSegment(p,zero,primitiveEndpoint(U_start_angle,U_radius),0.000001),
        primitiveSegment(p,zero,primitiveEndpoint(U_end_angle,U_radius),0.000001));
    o[0]=primitiveInsideAngle(p) && radial.v<=U_radius.v ? neg(distance) : distance;
}
"""
PROGRAM = ShaderProgram("sector", BODY)
