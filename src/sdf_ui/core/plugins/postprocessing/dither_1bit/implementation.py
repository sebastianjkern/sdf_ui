"""One-bit dithering's shader interface and true zero-gradient rule."""

from sdf_ui.core.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.core.plugins.base import ImageBinding, ShaderInterface
from sdf_ui.core.plugins.layer.glsl import color_program
from sdf_ui.core.shader_expression import ShaderProgram
from ..bayer import GLSL, threshold


def reference(context, inputs, params):
    import numpy as np
    from sdf_ui.learning.render import _rgba, _tensor

    color = _tensor(inputs["origTex"])
    luminance = np.sum(color.data[..., :3] * (0.299, 0.587, 0.114), axis=-1)
    step = (luminance >= threshold(context)).astype(float)
    return _rgba(color[..., :3] * 0.0 + step[..., None], color[..., 3] * 0.0 + 1.0)


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("origTex", 1, read=True),)
)
CONTRACT = ShaderDifferentiation(
    "dither_1bit", "core/plugins/postprocessing/dither_1bit/shader.glsl",
    (ShaderArgument("origTex", (4,), "rgba8"),), (),
    (ShaderArgument("destTex", (4,), "rgba8"),), behavior="zero_almost_everywhere",
    notes="True zero derivative almost everywhere; undefined at thresholds. No straight-through estimator.",
    reference=reference,
)
BODY = GLSL + color_program(r"""
    D4 source=load4(0,pos);
    float luminance=source.x.v*0.299+source.y.v*0.587+source.z.v*0.114;
    D bit=c(luminance<effectsThreshold(pos)?0.0:1.0);
    D4 col=D4(bit,bit,bit,c(1.0));
""")
PROGRAM = ShaderProgram("dither_1bit", BODY)
