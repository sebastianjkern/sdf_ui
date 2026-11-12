"""Film-grain shader layout and fixed-coordinate differentiation rules."""

from sdf_ui.core.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.core.plugins.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.core.shader_expression import ShaderProgram
from ..glsl import RGBA


def reference(context, inputs, params):
    import numpy as np
    from sdf_ui.learning.autodiff import Tensor

    x, y = context.x.data.astype(np.float32), context.y.data.astype(np.float32)
    # GLSL's dot product rounds once after the sum (commonly via FMA). Match
    # that float32 result before evaluating its implementation-dependent cos.
    angle = (
        x.astype(np.float64) * np.float32(23.14069263277926)
        + y.astype(np.float64) * np.float32(2.665144142690225)
    ).astype(np.float32)
    value = np.cos(angle) * np.float32(12345.6789)
    value = value - np.floor(value)
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
float shadingNoise(vec2 p) {
    vec2 K1 = vec2(23.14069263277926, 2.665144142690225);
    return fract(cos(dot(p, K1)) * 12345.6789);
}
void evaluate(ivec2 pos, out D o[8]) {
    // Pixel coordinates are fixed and the native time uniform is unused.
    D grain = c(shadingNoise(vec2(pos)));
    shadingOutput(D4(grain, grain, grain, c(1.0)), o);
}
"""
PROGRAM = ShaderProgram("film_grain", RGBA + BODY)
