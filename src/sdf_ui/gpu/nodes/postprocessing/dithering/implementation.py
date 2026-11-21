"""Ordered dithering's shader interface and differentiable implementation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)

from ..bayer import threshold


def reference(context, inputs, params):
    from sdf_ui.learning.render import _rgba, _tensor

    color = _tensor(inputs["origTex"])
    step = (color.data[..., :3] >= threshold(context)[..., None]) / 255.0
    return _rgba(color[..., :3] + step, color[..., 3])


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("origTex", 1, read=True),)
)
CONTRACT = ShaderDifferentiation(
    "dithering", "gpu/nodes/postprocessing/dithering/dithering.wgsl",
    (ShaderArgument("origTex", (4,), "rgba8"),), (),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Additive threshold derivative zero; original color derivative identity away from jumps.",
    reference=reference,
)
