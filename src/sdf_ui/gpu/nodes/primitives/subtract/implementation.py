"""Subtraction SDF shader metadata and differentiation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import maximum
    return maximum(inputs["sdf0"], -inputs["sdf1"])


INTERFACE = ShaderInterface(ImageBinding("destTex", 0, write=True), (ImageBinding("sdf0", 1, read=True), ImageBinding("sdf1", 2, read=True)))
CONTRACT = ShaderDifferentiation(
    "subtract", "gpu/nodes/primitives/subtract/subtract.wgsl",
    (ShaderArgument("sdf0", (), "r32f"), ShaderArgument("sdf1", (), "r32f")), (),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
