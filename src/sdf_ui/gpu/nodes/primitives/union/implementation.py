"""Union SDF shader metadata and differentiation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import minimum
    return minimum(inputs["sdf0"], inputs["sdf1"])


INTERFACE = ShaderInterface(ImageBinding("destTex", 0, write=True), (ImageBinding("sdf0", 1, read=True), ImageBinding("sdf1", 2, read=True)))
CONTRACT = ShaderDifferentiation(
    "union", "gpu/nodes/primitives/union/union.wgsl",
    (ShaderArgument("sdf0", (), "r32f"), ShaderArgument("sdf1", (), "r32f")), (),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
