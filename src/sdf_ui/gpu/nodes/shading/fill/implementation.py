"""Fill shader's reference math and differentiable program."""

from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)
from sdf_ui.gpu.nodes.shader_expression import ShaderProgram

from ..glsl import RGBA


def reference(context, inputs, params):
    from sdf_ui.learning.render import _mix, _rgb_to_lab, _smoothstep, _tensor
    from ..sampling import sample

    distance = inputs["sdf"] - _tensor(params["inflate"])
    dx = sample(inputs["sdf"], 1, 0) - sample(inputs["sdf"], -1, 0)
    dy = sample(inputs["sdf"], 0, 1) - sample(inputs["sdf"], 0, -1)
    edge_width = (dx.abs() + dy.abs()) * 0.25
    edge_width = edge_width + 0.0001
    lower = _tensor(params["second"]) - edge_width
    from sdf_ui.learning.autodiff import maximum

    lower = maximum(lower, _tensor(params["first"]))
    upper = _tensor(params["second"]) + edge_width
    weight = _smoothstep(lower, upper, distance)
    return _mix(
        _rgb_to_lab(_tensor(params["color"])),
        _rgb_to_lab(_tensor(params["background"])),
        weight[..., None],
    )


CONTRACT = ShaderDifferentiation(
    "fill",
    "gpu/nodes/shading/fill/shader.glsl",
    (ShaderArgument("sdf", (), "r32f"),),
    (
        ShaderArgument("inflate", (), "float"),
        ShaderArgument("color", (4,), "vec4"),
        ShaderArgument("background", (4,), "vec4"),
        ShaderArgument("first", (), "float"),
        ShaderArgument("second", (), "float"),
    ),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Scale-aware one-pixel edge smoothing; min/max ties split adjoints equally.",
    reference=reference,
)

BODY = """
void evaluate(ivec2 pos, out D o[8]) {
    D distance = sub(ld(0, pos, 0), U_inflate);
    D dx = sub(ld(0, clamp(pos + ivec2(1, 0), ivec2(0), pixelSize - ivec2(1)), 0),
               ld(0, clamp(pos - ivec2(1, 0), ivec2(0), pixelSize - ivec2(1)), 0));
    D dy = sub(ld(0, clamp(pos + ivec2(0, 1), ivec2(0), pixelSize - ivec2(1)), 0),
               ld(0, clamp(pos - ivec2(0, 1), ivec2(0), pixelSize - ivec2(1)), 0));
    D edgeWidth = add(mul(add(absD(dx), absD(dy)), c(0.25)), c(0.0001));
    D lower = maxD(U_first, sub(U_second, edgeWidth));
    D upper = add(U_second, edgeWidth);
    D weight = smoothD(lower, upper, distance);
    shadingOutput(shadingMix(rgbToLab(U_color), rgbToLab(U_background), weight), o);
}
"""
PROGRAM = ShaderProgram("fill", RGBA + BODY)
