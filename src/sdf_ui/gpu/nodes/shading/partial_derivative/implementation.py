"""Finite-difference shader's image layout and differentiable rules."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)

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
    "partial_derivative", "gpu/nodes/shading/partial_derivative/partial_derivative.wgsl",
    (ShaderArgument("sdf", (), "r32f"),), (),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    behavior="smooth_for_fixed_sampling",
    notes="Transpose of the forward finite-difference stencil, including border reads.",
    sampling="integer_zero_outside", reference=reference,
)
