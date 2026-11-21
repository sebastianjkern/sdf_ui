"""Rectangle shader's uniform layout and differentiable implementation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)

from ..rotation import rotate


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import maximum, minimum
    from sdf_ui.learning.render import _length, _mix, _smoothstep, _tensor

    offset = _tensor(params["offset"])
    x, y = rotate(context.x - offset[0], context.y - offset[1], _tensor(params["angle"]))
    size, radius = _tensor(params["size"]), _tensor(params["corner_radius"])
    if radius.shape == (4,):
        hx, hy = _smoothstep(0.0, 0.0001, x), _smoothstep(0.0, 0.0001, y)
        radius = _mix(_mix(radius[0], radius[2], hx), _mix(radius[1], radius[3], hx), hy)
    elif radius.shape != ():
        raise ValueError("corner_radius must be a scalar or vec4")
    qx, qy = x.abs() - size[0] + radius, y.abs() - size[1] + radius
    return minimum(maximum(qx, qy), 0.0) + _length(maximum(qx, 0.0), maximum(qy, 0.0)) - radius


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    uniforms=(Uniform("size", "vec2"), Uniform("offset", "vec2"), Uniform("corner_radius", "vec4"), Uniform("angle", "float")),
)
CONTRACT = ShaderDifferentiation(
    "rect", "gpu/nodes/primitives/rect/rect.wgsl", (),
    (ShaderArgument("size", (2,), "vec2"), ShaderArgument("offset", (2,), "vec2"), ShaderArgument("corner_radius", (4,), "vec4"), ShaderArgument("angle", (), "float")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
