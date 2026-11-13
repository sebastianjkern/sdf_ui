"""Invert shader's image bindings and differentiable implementations."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)
from sdf_ui.gpu.nodes.layer.glsl import color_program
from sdf_ui.gpu.nodes.shader_expression import ShaderProgram


def reference(context, inputs, params):
    from sdf_ui.learning.render import _rgba, _tensor

    color = _tensor(inputs["origTex"])
    return _rgba(1.0 - color[..., :3], color[..., 3])


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("origTex", 1, read=True),)
)
CONTRACT = ShaderDifferentiation(
    "invert", "gpu/nodes/postprocessing/invert/shader.glsl",
    (ShaderArgument("origTex", (4,), "rgba8"),), (),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    behavior="smooth_for_fixed_sampling",
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
BODY = color_program(r"""
    D4 source=load4(0,pos);
    D4 col=D4(sub(c(1.0),source.x),sub(c(1.0),source.y),
              sub(c(1.0),source.z),source.w);
""")
PROGRAM = ShaderProgram("invert", BODY)
