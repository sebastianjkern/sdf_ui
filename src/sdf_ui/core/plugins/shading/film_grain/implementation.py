"""Film-grain shader layout and fixed-coordinate differentiation rules."""

from sdf_ui.core.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.core.plugins.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.core.shader_expression import ShaderProgram
from ..glsl import RGBA


def reference(context, inputs, params):
    import numpy as np
    from sdf_ui.learning.autodiff import Tensor

    x, y = context.x.data.astype(np.uint32), context.y.data.astype(np.uint32)
    value = x * np.uint32(0x9E3779B9) + y * np.uint32(0x85EBCA6B)
    value ^= value >> np.uint32(16)
    value *= np.uint32(0x7FEB352D)
    value ^= value >> np.uint32(15)
    value *= np.uint32(0x846CA68B)
    value ^= value >> np.uint32(16)
    value = (value & np.uint32(0x00FFFFFF)).astype(np.float64) / 16777216.0
    return Tensor(np.stack((value, value, value, np.ones_like(value)), axis=-1))


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), uniforms=(Uniform("time", "float"),),
)
CONTRACT = ShaderDifferentiation(
    "film_grain", "core/plugins/shading/film_grain/shader.glsl", (),
    (ShaderArgument("time", (), "float"),), (ShaderArgument("destTex", (4,), "rgba8"),),
    behavior="constant", zero_gradient_parameters=("time",),
    notes="Fixed pixel coordinates/hash constants; declared time uniform is unused and has zero derivative.",
    reference=reference,
)
BODY = """
float shadingNoise(uvec2 p) {
    uint value = p.x * 0x9E3779B9u + p.y * 0x85EBCA6Bu;
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    value ^= value >> 16;
    return float(value & 0x00FFFFFFu) / 16777216.0;
}
void evaluate(ivec2 pos, out D o[8]) {
    // Integer hashing avoids renderer-dependent large-argument cosine results.
    D grain = c(shadingNoise(uvec2(pos)));
    shadingOutput(D4(grain, grain, grain, c(1.0)), o);
}
"""
PROGRAM = ShaderProgram("film_grain", RGBA + BODY)
