"""Masked union's two-output shader interface and differentiation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)
from sdf_ui.gpu.nodes.shader_expression import ShaderProgram


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import minimum, stack
    from sdf_ui.learning.render import _smoothstep

    distance = minimum(inputs["sdf0"], inputs["sdf1"])
    mask = _smoothstep(0.0, 1.0, inputs["sdf1"] - distance)
    return distance, stack((mask, mask, mask, mask * 0.0 + 1.0), axis=-1)


INTERFACE = ShaderInterface(
    output=ImageBinding("destTex", 0, write=True),
    extra_outputs=(ImageBinding("maskTex", 1, write=True),),
    inputs=(ImageBinding("sdf0", 2, read=True), ImageBinding("sdf1", 3, read=True)),
)
CONTRACT = ShaderDifferentiation(
    "masked_union", "gpu/nodes/primitives/masked_union/shader.glsl",
    (ShaderArgument("sdf0", (), "r32f"), ShaderArgument("sdf1", (), "r32f")), (),
    (ShaderArgument("destTex", (), "r32f"), ShaderArgument("maskTex", (4,), "rgba8")),
    notes="Distance and mask adjoints both contribute to source fields.", reference=reference,
)
BODY = r"""
void evaluate(ivec2 pos,out D o[8]) {
    D a=ld(0,pos,0), b=ld(1,pos,0);
    o[0]=minD(a,b);
    D mask=smoothD(c(0.0),c(1.0),sub(b,o[0]));
    o[1]=mask; o[2]=mask; o[3]=mask; o[4]=c(1.0);
}
"""
PROGRAM = ShaderProgram("masked_union", BODY)
