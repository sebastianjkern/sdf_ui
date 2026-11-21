"""Invert shader's image bindings and differentiable implementations."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


def reference(context, inputs, params):
    from sdf_ui.learning.render import _rgba, _tensor

    color = _tensor(inputs["origTex"])
    return _rgba(1.0 - color[..., :3], color[..., 3])


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("origTex", 1, read=True),)
)
CONTRACT = ShaderDifferentiation(
    "invert", "gpu/nodes/postprocessing/invert/invert.wgsl",
    (ShaderArgument("origTex", (4,), "rgba8"),), (),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    behavior="smooth_for_fixed_sampling",
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
