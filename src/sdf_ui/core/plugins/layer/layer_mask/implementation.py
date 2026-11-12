"""Layer-mask extension's image interface and differentiable implementation."""

from sdf_ui.core.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.core.plugins.base import ImageBinding, ShaderInterface
from sdf_ui.core.shader_expression import ShaderProgram
from ..glsl import color_program


def reference(context, inputs, params):
    from sdf_ui.learning.render import _tensor

    first, second = _tensor(inputs["tex0"]), _tensor(inputs["tex1"])
    alpha = _tensor(inputs["mask"])[..., 0:1]
    return second * (1.0 - alpha) + first * alpha


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    (ImageBinding("tex0", 1, read=True), ImageBinding("tex1", 2, read=True), ImageBinding("mask", 3, read=True)),
)
CONTRACT = ShaderDifferentiation(
    "layer_mask", "core/plugins/layer/layer_mask/shader.glsl",
    tuple(ShaderArgument(name, (4,), "rgba8") for name in ("tex0", "tex1", "mask")),
    (), (ShaderArgument("destTex", (4,), "rgba8"),), behavior="smooth_for_fixed_sampling",
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
BODY = color_program(r"""
    D4 first=load4(0,pos), second=load4(1,pos);
    D alpha=ld(2,pos,0);
    D4 col=mix4(second,first,alpha);
""")
PROGRAM = ShaderProgram("layer_mask", BODY)
