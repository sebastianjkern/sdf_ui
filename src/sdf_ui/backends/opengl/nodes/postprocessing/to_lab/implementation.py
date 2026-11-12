"""RGB-to-LAB shader's image bindings and differentiable implementations."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.backends.opengl.nodes.layer.glsl import color_program
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram


def reference(context, inputs, params):
    from sdf_ui.learning.render import _rgb_to_lab, _tensor

    return _rgb_to_lab(_tensor(inputs["origTex"]))


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("origTex", 1, read=True),)
)
CONTRACT = ShaderDifferentiation(
    "to_lab", "backends/opengl/nodes/postprocessing/to_lab/shader.glsl",
    (ShaderArgument("origTex", (4,), "rgba8"),), (),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
BODY = color_program("D4 col=rgbToLab(load4(0,pos));")
PROGRAM = ShaderProgram("to_lab", BODY)
