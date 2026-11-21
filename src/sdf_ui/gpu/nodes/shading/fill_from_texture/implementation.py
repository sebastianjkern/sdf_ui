"""Texture-fill shader interface, reference evaluation and GPU program."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


def reference(context, inputs, params):
    from sdf_ui.learning.render import _mix, _smoothstep, _tensor

    distance = inputs["sdf"] - _tensor(params["inflate"])
    weight = _smoothstep(0.0, 1.5001, distance)
    return _mix(inputs["origTex"], _tensor(params["background"]), weight[..., None])


INTERFACE = ShaderInterface(
    output=ImageBinding("destTex", 0, write=True),
    inputs=(ImageBinding("origTex", 1, read=True), ImageBinding("sdf", 2, read=True)),
    uniforms=(Uniform("inflate", "float"), Uniform("background", "vec4")),
)
CONTRACT = ShaderDifferentiation(
    "fill_from_texture", "gpu/nodes/shading/fill_from_texture/fill_from_texture.wgsl",
    (ShaderArgument("origTex", (4,), "rgba8"), ShaderArgument("sdf", (), "r32f")),
    (ShaderArgument("inflate", (), "float"), ShaderArgument("background", (4,), "vec4")),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.",
    reference=reference,
)
