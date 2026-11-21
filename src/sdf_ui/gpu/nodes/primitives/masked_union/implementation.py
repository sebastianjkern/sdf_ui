"""Masked union's two-output shader interface and differentiation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import maximum, minimum, stack
    from sdf_ui.learning.render import _smoothstep, _tensor

    distance = minimum(inputs["sdf0"], inputs["sdf1"])
    width = maximum(abs(_tensor(params.get("edgeWidth", 0.5))), 0.0001)
    mask = _smoothstep(-width, width, inputs["sdf1"] - inputs["sdf0"])
    return distance, stack((mask, mask, mask, mask * 0.0 + 1.0), axis=-1)


INTERFACE = ShaderInterface(
    output=ImageBinding("destTex", 0, write=True),
    extra_outputs=(ImageBinding("maskTex", 1, write=True),),
    inputs=(ImageBinding("sdf0", 2, read=True), ImageBinding("sdf1", 3, read=True)),
    uniforms=(Uniform("edgeWidth", "float"),),
)
CONTRACT = ShaderDifferentiation(
    "masked_union",
    "gpu/nodes/primitives/masked_union/masked_union.wgsl",
    (ShaderArgument("sdf0", (), "r32f"), ShaderArgument("sdf1", (), "r32f")),
    (ShaderArgument("edgeWidth", (), "float"),),
    (ShaderArgument("destTex", (), "r32f"), ShaderArgument("maskTex", (4,), "rgba8")),
    notes="Distance and mask adjoints both contribute to source fields.",
    reference=reference,
    defaults={"edgeWidth": 0.5},
)
