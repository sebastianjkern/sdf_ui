"""Half-plane shader interface and differentiation."""

from sdf_ui.core.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.core.plugins.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.core.shader_expression import ShaderProgram


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import maximum
    from sdf_ui.learning.render import _length, _tensor

    point, normal = _tensor(params["point"]), _tensor(params["normal"])
    x, y = context.x - point[0], context.y - point[1]
    norm = maximum(_length(normal[0], normal[1]), 1e-6)
    return (x * normal[0] + y * normal[1]) / norm


INTERFACE = ShaderInterface(ImageBinding("destTex", 0, write=True), uniforms=(Uniform("point", "vec2"), Uniform("normal", "vec2")))
CONTRACT = ShaderDifferentiation(
    "half_plane", "core/plugins/primitives/half_plane/shader.glsl", (),
    (ShaderArgument("point", (2,), "vec2"), ShaderArgument("normal", (2,), "vec2")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
BODY = r"""
void evaluate(ivec2 pos,out D o[8]) {
    o[0]=divD(dot2(sub2(v2(vec2(pos)),U_point),U_normal),
              maxD(len2(U_normal),c(0.000001)));
}
"""
PROGRAM = ShaderProgram("half_plane", BODY)
