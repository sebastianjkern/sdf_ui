"""Ring shader's declared interface and differentiable implementation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


def reference(context, inputs, params):
    from sdf_ui.learning.render import _length, _tensor

    center = _tensor(params["center"])
    distance = _length(context.x - center[0], context.y - center[1])
    return (distance - _tensor(params["radius"])).abs() - _tensor(params["thickness"]) * 0.5


INTERFACE = ShaderInterface(
    output=ImageBinding("destTex", 0, write=True),
    uniforms=(Uniform("center", "vec2"), Uniform("radius", "float"), Uniform("thickness", "float")),
)
CONTRACT = ShaderDifferentiation(
    "ring", "gpu/nodes/primitives/ring/ring.wgsl", (),
    (ShaderArgument("center", (2,), "vec2"), ShaderArgument("radius", (), "float"), ShaderArgument("thickness", (), "float")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
