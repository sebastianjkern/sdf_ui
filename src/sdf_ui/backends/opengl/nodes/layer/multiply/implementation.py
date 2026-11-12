"""Multiply extension's image interface and differentiable implementation."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram
from ..glsl import color_program


def reference(context, inputs, params):
    from sdf_ui.learning.render import _tensor

    return _tensor(inputs["mask1"]) * _tensor(inputs["mask2"])


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    (ImageBinding("mask1", 1, read=True), ImageBinding("mask2", 2, read=True)),
)
CONTRACT = ShaderDifferentiation(
    "multiply", "backends/opengl/nodes/layer/multiply/shader.glsl",
    (ShaderArgument("mask1", (4,), "rgba8"), ShaderArgument("mask2", (4,), "rgba8")),
    (), (ShaderArgument("destTex", (4,), "rgba8"),),
    behavior="smooth_for_fixed_sampling",
    notes="Piecewise derivatives; min/max ties split adjoints equally.",
    reference=reference,
)
BODY = color_program(r"""
    D4 first=load4(0,pos), second=load4(1,pos);
    D4 col=D4(mul(first.x,second.x),mul(first.y,second.y),
              mul(first.z,second.z),mul(first.w,second.w));
""")
PROGRAM = ShaderProgram("multiply", BODY)
