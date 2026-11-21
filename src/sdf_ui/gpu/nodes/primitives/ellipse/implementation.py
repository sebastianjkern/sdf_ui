"""Ellipse shader's uniform interface and differentiable implementation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import maximum
    from sdf_ui.learning.render import _length, _tensor

    center = _tensor(params["center"])
    x, y = context.x - center[0], context.y - center[1]
    radii = maximum(_tensor(params["radii"]).abs(), 1e-6)
    f = (x / radii[0]) ** 2 + (y / radii[1]) ** 2 - 1.0
    return f / maximum(_length(2.0 * x / radii[0] ** 2, 2.0 * y / radii[1] ** 2), 1e-6)


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    uniforms=(Uniform("center", "vec2"), Uniform("radii", "vec2")),
)
CONTRACT = ShaderDifferentiation(
    "ellipse", "gpu/nodes/primitives/ellipse/ellipse.wgsl", (),
    (ShaderArgument("center", (2,), "vec2"), ShaderArgument("radii", (2,), "vec2")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
