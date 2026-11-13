"""Isoline shader's layout and differentiable color-band rules."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)
from sdf_ui.gpu.nodes.shader_expression import ShaderProgram

from ..glsl import RGBA


def reference(context, inputs, params):
    import numpy as np

    from sdf_ui.learning.autodiff import maximum, minimum, where
    from sdf_ui.learning.render import _mix, _rgb_to_lab, _smoothstep, _tensor

    distance = abs(inputs["sdf"] - _tensor(params["inflate"]))
    valid = distance.data <= 1e20
    distance = where(valid, distance, 0.0)
    period = maximum(_tensor(params["spacing"]), 1e-6)
    shifted = distance + _tensor(params["phase"])
    wrapped = shifted - np.floor((shifted / period).data) * period
    wrapped = where(wrapped.data < 0.0, wrapped + period, wrapped)
    band = minimum(wrapped, period - wrapped)
    half_width = maximum(_tensor(params["line_width"]) * 0.5, 0.0)
    aa = 1.5001 + maximum(_tensor(params["feather"]), 0.0)
    line = 1.0 - _smoothstep(maximum(half_width - aa, 0.0), half_width + aa, band)
    line = where(valid, line, 0.0)
    return _mix(
        _rgb_to_lab(_tensor(params["background"])),
        _rgb_to_lab(_tensor(params["line_color"])), line[..., None],
    )


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("sdf", 1, read=True),),
    (Uniform("background", "vec4"), Uniform("line_color", "vec4"),
     Uniform("inflate", "float"), Uniform("spacing", "float"),
     Uniform("line_width", "float"), Uniform("feather", "float"), Uniform("phase", "float")),
)
CONTRACT = ShaderDifferentiation(
    "isolines", "gpu/nodes/shading/isolines/shader.glsl",
    (ShaderArgument("sdf", (), "r32f"),),
    (ShaderArgument("background", (4,), "vec4"), ShaderArgument("line_color", (4,), "vec4"),
     ShaderArgument("inflate", (), "float"), ShaderArgument("spacing", (), "float"),
     ShaderArgument("line_width", (), "float"), ShaderArgument("feather", (), "float"),
     ShaderArgument("phase", (), "float")),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
BODY = """
void evaluate(ivec2 pos, out D o[8]) {
    D distance = absD(sub(ld(0, pos, 0), U_inflate));
    D4 background = rgbToLab(U_background);
    if (isnan(distance.v) || distance.v > 1e20) {
        shadingOutput(background, o);
        return;
    }
    D period = maxD(U_spacing, c(1e-6));
    D wrapped = modD(add(distance, U_phase), period);
    if (wrapped.v < 0.0) wrapped = add(wrapped, period);
    D band = minD(wrapped, sub(period, wrapped));
    D halfWidth = maxD(mul(U_line_width, c(0.5)), c(0.0));
    D aa = add(c(1.5001), maxD(U_feather, c(0.0)));
    D weight = sub(c(1.0), smoothD(maxD(sub(halfWidth, aa), c(0.0)),
                                  add(halfWidth, aa), band));
    shadingOutput(shadingMix(background, rgbToLab(U_line_color), weight), o);
}
"""
PROGRAM = ShaderProgram("isolines", RGBA + BODY)
