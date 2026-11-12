"""Fill shader's reference math and differentiable OpenGL program."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram
from ..glsl import RGBA


def reference(context, inputs, params):
    from sdf_ui.learning.render import _mix, _rgb_to_lab, _smoothstep, _tensor

    distance = inputs["sdf"] - _tensor(params["inflate"])
    weight = _smoothstep(
        _tensor(params["first"]), _tensor(params["second"]) + 0.0001, distance
    )
    return _mix(
        _rgb_to_lab(_tensor(params["color"])),
        _rgb_to_lab(_tensor(params["background"])),
        weight[..., None],
    )


CONTRACT = ShaderDifferentiation(
    "fill", "backends/opengl/nodes/shading/fill/shader.glsl",
    (ShaderArgument("sdf", (), "r32f"),),
    (
        ShaderArgument("inflate", (), "float"),
        ShaderArgument("color", (4,), "vec4"),
        ShaderArgument("background", (4,), "vec4"),
        ShaderArgument("first", (), "float"),
        ShaderArgument("second", (), "float"),
    ),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.",
    reference=reference,
)

BODY = """
void evaluate(ivec2 pos, out D o[8]) {
    D distance = sub(ld(0, pos, 0), U_inflate);
    D weight = smoothD(U_first, add(U_second, c(0.0001)), distance);
    shadingOutput(shadingMix(rgbToLab(U_color), rgbToLab(U_background), weight), o);
}
"""
PROGRAM = ShaderProgram("fill", RGBA + BODY)
