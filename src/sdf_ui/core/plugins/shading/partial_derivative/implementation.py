"""Finite-difference shader's image layout and differentiable rules."""

from sdf_ui.core.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.core.plugins.base import ImageBinding, ShaderInterface
from sdf_ui.core.shader_expression import ShaderProgram
from ..glsl import RGBA
from ..sampling import sample


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import stack

    sdf = inputs["sdf"]
    zero = sdf * 0.0
    return stack(
        (
            sdf - sample(sdf, 1, 0, zero=True),
            sdf - sample(sdf, 0, 1, zero=True),
            zero,
            zero + 1.0,
        ),
        axis=-1,
    )


INTERFACE = ShaderInterface(
    output=ImageBinding("destTex", 0, write=True),
    inputs=(ImageBinding("sdf", 1, read=True),),
)
CONTRACT = ShaderDifferentiation(
    "partial_derivative", "core/plugins/shading/partial_derivative/shader.glsl",
    (ShaderArgument("sdf", (), "r32f"),), (),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    behavior="smooth_for_fixed_sampling",
    notes="Transpose of the forward finite-difference stencil, including border reads.",
    sampling="integer_zero_outside", reference=reference,
)
BODY = """
void evaluate(ivec2 pos, out D o[8]) {
    D center = ld(0, pos, 0);
    shadingOutput(D4(sub(center, ld(0, pos + ivec2(1, 0), 0)),
                     sub(center, ld(0, pos + ivec2(0, 1), 0)),
                     c(0.0), c(1.0)), o);
}
"""
PROGRAM = ShaderProgram("partial_derivative", RGBA + BODY)
