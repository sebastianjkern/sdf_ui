"""Arc shader's uniform interface and differentiable implementation."""



from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)

from ..angular import angles

_FLOAT_TEXT = 'float'

def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import minimum, where
    from sdf_ui.learning.render import _length, _tensor

    center = _tensor(params["center"])
    x, y = context.x - center[0], context.y - center[1]
    start, end, inside = angles(x, y, params)
    radius = _tensor(params["radius"])
    endpoints = minimum(
        _length(x - radius * start.cos(), y - radius * start.sin()),
        _length(x - radius * end.cos(), y - radius * end.sin()),
    )
    return where(inside, (_length(x, y) - radius).abs(), endpoints)


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    uniforms=(Uniform("center", "vec2"), Uniform("radius", _FLOAT_TEXT),
              Uniform("start_angle", _FLOAT_TEXT), Uniform("end_angle", _FLOAT_TEXT)),
)
CONTRACT = ShaderDifferentiation(
    "arc", "gpu/nodes/primitives/arc/arc.wgsl", (),
    (ShaderArgument("center", (2,), "vec2"), ShaderArgument("radius", (), _FLOAT_TEXT),
     ShaderArgument("start_angle", (), _FLOAT_TEXT), ShaderArgument("end_angle", (), _FLOAT_TEXT)),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Active radial or endpoint branch; angular selection is fixed away from boundaries.",
    reference=reference,
)
