"""Arc shader's uniform interface and differentiable implementation."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram
from ..angular import GLSL, angles


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import minimum, where
    from sdf_ui.learning.render import _length, _tensor

    center = _tensor(params["center"])
    x, y = context.x - center[0], context.y - center[1]
    start, end, inside = angles(x, y, params)
    radius = _tensor(params["radius"])
    endpoints = minimum(
        _length(x - radius * start.cos(), y - radius * start.sin()),
        _length(x - radius * end.cos(), y - radius * end.sin()),
    )
    return where(inside, (_length(x, y) - radius).abs(), endpoints)


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    uniforms=(Uniform("center", "vec2"), Uniform("radius", "float"),
              Uniform("start_angle", "float"), Uniform("end_angle", "float")),
)
CONTRACT = ShaderDifferentiation(
    "arc", "backends/opengl/nodes/primitives/arc/shader.glsl", (),
    (ShaderArgument("center", (2,), "vec2"), ShaderArgument("radius", (), "float"),
     ShaderArgument("start_angle", (), "float"), ShaderArgument("end_angle", (), "float")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Active radial or endpoint branch; angular selection is fixed away from boundaries.",
    reference=reference,
)
BODY = GLSL + r"""
void evaluate(ivec2 pos,out D o[8]) {
    D2 p=sub2(v2(vec2(pos)),U_center);
    if(primitiveInsideAngle(p)) o[0]=absD(sub(len2(p),U_radius));
    else o[0]=minD(len2(sub2(p,primitiveEndpoint(U_start_angle,U_radius))),
                   len2(sub2(p,primitiveEndpoint(U_end_angle,U_radius))));
}
"""
PROGRAM = ShaderProgram("arc", BODY)
