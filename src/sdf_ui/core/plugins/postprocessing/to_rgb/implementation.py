"""LAB-to-RGB shader's image bindings and differentiable implementations."""

from sdf_ui.core.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.core.plugins.base import ImageBinding, ShaderInterface
from sdf_ui.core.plugins.layer.glsl import color_program
from sdf_ui.core.shader_expression import ShaderProgram


def reference(context, inputs, params):
    from sdf_ui.learning.render import _lab_to_rgb, _tensor

    return _lab_to_rgb(_tensor(inputs["origTex"]))


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("origTex", 1, read=True),)
)
CONTRACT = ShaderDifferentiation(
    "to_rgb", "core/plugins/postprocessing/to_rgb/shader.glsl",
    (ShaderArgument("origTex", (4,), "rgba8"),), (),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
BODY = color_program("D4 col=labToRgb(load4(0,pos));")
PROGRAM = ShaderProgram("to_rgb", BODY)
