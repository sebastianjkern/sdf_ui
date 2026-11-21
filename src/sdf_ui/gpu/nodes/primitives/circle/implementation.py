"""Circle's unquantized reference and shader differentiation information."""

from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


def reference(context, inputs, params):
    from sdf_ui.learning.render import _length, _tensor

    offset = _tensor(params["offset"])
    return _length(context.x - offset[0], context.y - offset[1]) - _tensor(params["radius"])


CONTRACT = ShaderDifferentiation(
    "circle",
    "gpu/nodes/primitives/circle/circle.wgsl",
    (),
    (ShaderArgument("offset", (2,), "vec2"), ShaderArgument("radius", (), "float")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Radial distance; zero subgradient at the center.",
    reference=reference,
)


