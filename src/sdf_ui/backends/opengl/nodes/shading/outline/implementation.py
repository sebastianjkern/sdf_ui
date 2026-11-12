"""Outline shader interface, reference evaluation and GPU program."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram
from ..glsl import RGBA


def reference(context, inputs, params):
    from sdf_ui.learning.render import _mix, _rgb_to_lab, _smoothstep, _tensor

    distance = abs(inputs["sdf"] - _tensor(params["inflate"]))
    weight = _smoothstep(0.0, 1.5001, distance)
    return _mix(
        _rgb_to_lab(_tensor(params["outline"])),
        _rgb_to_lab(_tensor(params["background"])),
        weight[..., None],
    )


INTERFACE = ShaderInterface(
    output=ImageBinding("destTex", 0, write=True),
    inputs=(ImageBinding("sdf", 1, read=True),),
    uniforms=(Uniform("background", "vec4"), Uniform("outline", "vec4"), Uniform("inflate", "float")),
)
CONTRACT = ShaderDifferentiation(
    "outline", "backends/opengl/nodes/shading/outline/shader.glsl",
    (ShaderArgument("sdf", (), "r32f"),),
    (ShaderArgument("background", (4,), "vec4"), ShaderArgument("outline", (4,), "vec4"), ShaderArgument("inflate", (), "float")),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.",
    reference=reference,
)
BODY = """
void evaluate(ivec2 pos, out D o[8]) {
    D distance = absD(sub(ld(0, pos, 0), U_inflate));
    // Native OUTLINE_DISTANCE / 2 is integer division, giving zero.
    D weight = smoothD(c(0.0), c(1.5001), distance);
    shadingOutput(shadingMix(rgbToLab(U_outline), rgbToLab(U_background), weight), o);
}
"""
PROGRAM = ShaderProgram("outline", RGBA + BODY)
