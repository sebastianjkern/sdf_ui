"""Ordered dithering's shader interface and differentiable implementation."""

from sdf_ui.core.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.core.plugins.base import ImageBinding, ShaderInterface
from sdf_ui.core.plugins.layer.glsl import color_program
from sdf_ui.core.shader_expression import ShaderProgram
from ..bayer import GLSL, threshold


def reference(context, inputs, params):
    from sdf_ui.learning.render import _rgba, _tensor

    color = _tensor(inputs["origTex"])
    step = (color.data[..., :3] >= threshold(context)[..., None]) / 255.0
    return _rgba(color[..., :3] + step, color[..., 3])


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("origTex", 1, read=True),)
)
CONTRACT = ShaderDifferentiation(
    "dithering", "core/plugins/postprocessing/dithering/shader.glsl",
    (ShaderArgument("origTex", (4,), "rgba8"),), (),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Additive threshold derivative zero; original color derivative identity away from jumps.",
    reference=reference,
)
BODY = GLSL + color_program(r"""
    D4 col=load4(0,pos);
    float threshold=effectsThreshold(pos);
    col.x=add(col.x,c(col.x.v<threshold?0.0:1.0/255.0));
    col.y=add(col.y,c(col.y.v<threshold?0.0:1.0/255.0));
    col.z=add(col.z,c(col.z.v<threshold?0.0:1.0/255.0));
""")
PROGRAM = ShaderProgram("dithering", BODY)
