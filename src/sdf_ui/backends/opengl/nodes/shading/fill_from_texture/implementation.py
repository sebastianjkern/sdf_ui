"""Texture-fill shader interface, reference evaluation and GPU program."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram
from ..glsl import RGBA


def reference(context, inputs, params):
    from sdf_ui.learning.render import _mix, _smoothstep, _tensor

    distance = inputs["sdf"] - _tensor(params["inflate"])
    weight = _smoothstep(0.0, 1.5001, distance)
    return _mix(inputs["origTex"], _tensor(params["background"]), weight[..., None])


INTERFACE = ShaderInterface(
    output=ImageBinding("destTex", 0, write=True),
    inputs=(ImageBinding("origTex", 1, read=True), ImageBinding("sdf", 2, read=True)),
    uniforms=(Uniform("inflate", "float"), Uniform("background", "vec4")),
)
CONTRACT = ShaderDifferentiation(
    "fill_from_texture", "backends/opengl/nodes/shading/fill_from_texture/shader.glsl",
    (ShaderArgument("origTex", (4,), "rgba8"), ShaderArgument("sdf", (), "r32f")),
    (ShaderArgument("inflate", (), "float"), ShaderArgument("background", (4,), "vec4")),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.",
    reference=reference,
)
BODY = """
void evaluate(ivec2 pos, out D o[8]) {
    D distance = sub(ld(1, pos, 0), U_inflate);
    D weight = smoothD(c(0.0), c(1.5001), distance);
    // This shader mixes the supplied texture/color directly, without LAB.
    shadingOutput(shadingMix(load4(0, pos), U_background, weight), o);
}
"""
PROGRAM = ShaderProgram("fill_from_texture", RGBA + BODY)
