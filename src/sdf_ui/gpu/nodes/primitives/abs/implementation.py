"""Absolute-value SDF shader metadata and differentiation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


def reference(context, inputs, params):
    return inputs["sdf0"].abs()


INTERFACE = ShaderInterface(ImageBinding("destTex", 0, write=True), (ImageBinding("sdf0", 1, read=True),))
CONTRACT = ShaderDifferentiation(
    "abs", "gpu/nodes/primitives/abs/abs.wgsl",
    (ShaderArgument("sdf0", (), "r32f"),), (), (ShaderArgument("destTex", (), "r32f"),),
    notes="Zero subgradient at zero.", reference=reference,
)
