"""Line-segment shader interface and differentiation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)

from ..segment_math import segment


def reference(context, inputs, params):
    from sdf_ui.learning.render import _tensor

    a, b = _tensor(params["a"]), _tensor(params["b"])
    return segment(context.x, context.y, a[0], a[1], b[0], b[1])


INTERFACE = ShaderInterface(ImageBinding("destTex", 0, write=True), uniforms=(Uniform("a", "vec2"), Uniform("b", "vec2")))
CONTRACT = ShaderDifferentiation(
    "line", "gpu/nodes/primitives/line/line.wgsl", (),
    (ShaderArgument("a", (2,), "vec2"), ShaderArgument("b", (2,), "vec2")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Closest point; coincident endpoints are outside the finite shader domain.", reference=reference,
)
