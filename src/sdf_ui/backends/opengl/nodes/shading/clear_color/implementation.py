"""Clear-color shader interface, reference evaluation and GPU program."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram
from ..glsl import RGBA


def reference(context, inputs, params):
    from sdf_ui.learning.render import _rgb_to_lab, _tensor

    return _rgb_to_lab(_tensor(params["color"])) + (context.x * 0.0)[..., None]


INTERFACE = ShaderInterface(
    output=ImageBinding("destTex", 0, write=True),
    uniforms=(Uniform("color", "vec4"),),
)
CONTRACT = ShaderDifferentiation(
    "clear_color", "backends/opengl/nodes/shading/clear_color/shader.glsl", (),
    (ShaderArgument("color", (4,), "vec4"),),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.",
    reference=reference,
)
BODY = """
void evaluate(ivec2 pos, out D o[8]) {
    shadingOutput(rgbToLab(U_color), o);
}
"""
PROGRAM = ShaderProgram("clear_color", RGBA + BODY)
