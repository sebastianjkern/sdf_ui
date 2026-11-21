"""Sector shader's uniform interface and differentiable implementation."""



from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)

from ..angular import angles
from ..segment_math import minimum_candidates, segment

_FLOAT_TEXT = 'float'

def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import where
    from sdf_ui.learning.render import _length, _tensor

    center = _tensor(params["center"])
    x, y = context.x - center[0], context.y - center[1]
    start, end, inside = angles(x, y, params)
    radius = _tensor(params["radius"])
    radial = _length(x, y)
    distance = minimum_candidates(
        (radial - radius).abs(),
        segment(x, y, 0.0, 0.0, radius * start.cos(), radius * start.sin(), 1e-6),
        segment(x, y, 0.0, 0.0, radius * end.cos(), radius * end.sin(), 1e-6),
    )
    return where(inside & (radial.data <= radius.data), -distance, distance)


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    uniforms=(Uniform("center", "vec2"), Uniform("radius", _FLOAT_TEXT),
              Uniform("start_angle", _FLOAT_TEXT), Uniform("end_angle", _FLOAT_TEXT)),
)
CONTRACT = ShaderDifferentiation(
    "sector", "gpu/nodes/primitives/sector/sector.wgsl", (),
    (ShaderArgument("center", (2,), "vec2"), ShaderArgument("radius", (), _FLOAT_TEXT),
     ShaderArgument("start_angle", (), _FLOAT_TEXT), ShaderArgument("end_angle", (), _FLOAT_TEXT)),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
