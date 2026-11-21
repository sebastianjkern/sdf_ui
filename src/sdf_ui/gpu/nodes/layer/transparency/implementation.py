"""Transparency extension's image interface and differentiable implementation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


def reference(context, inputs, params):
    from sdf_ui.learning.render import _rgba, _tensor

    color = _tensor(inputs["tex0"])
    return _rgba(color[..., :3], color[..., 3] * _tensor(params["alpha"]))


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    (ImageBinding("tex0", 1, read=True),), (Uniform("alpha", "float"),),
)
CONTRACT = ShaderDifferentiation(
    "transparency", "gpu/nodes/layer/transparency/transparency.wgsl",
    (ShaderArgument("tex0", (4,), "rgba8"),), (ShaderArgument("alpha", (), "float"),),
    (ShaderArgument("destTex", (4,), "rgba8"),), behavior="smooth_for_fixed_sampling",
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
