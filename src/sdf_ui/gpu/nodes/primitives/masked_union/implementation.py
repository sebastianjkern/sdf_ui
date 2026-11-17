"""Masked union's two-output shader interface and differentiation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)
from sdf_ui.gpu.nodes.shader_expression import ShaderProgram


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import maximum, minimum, stack
    from sdf_ui.learning.render import _smoothstep, _tensor

    distance = minimum(inputs["sdf0"], inputs["sdf1"])
    width = maximum(abs(_tensor(params.get("edgeWidth", 0.5))), 0.0001)
    mask = _smoothstep(-width, width, inputs["sdf1"] - inputs["sdf0"])
    return distance, stack((mask, mask, mask, mask * 0.0 + 1.0), axis=-1)


INTERFACE = ShaderInterface(
    output=ImageBinding("destTex", 0, write=True),
    extra_outputs=(ImageBinding("maskTex", 1, write=True),),
    inputs=(ImageBinding("sdf0", 2, read=True), ImageBinding("sdf1", 3, read=True)),
    uniforms=(Uniform("edgeWidth", "float"),),
)
CONTRACT = ShaderDifferentiation(
    "masked_union",
    "gpu/nodes/primitives/masked_union/shader.glsl",
    (ShaderArgument("sdf0", (), "r32f"), ShaderArgument("sdf1", (), "r32f")),
    (ShaderArgument("edgeWidth", (), "float"),),
    (ShaderArgument("destTex", (), "r32f"), ShaderArgument("maskTex", (4,), "rgba8")),
    notes="Distance and mask adjoints both contribute to source fields.",
    reference=reference,
    defaults={"edgeWidth": 0.5},
)
BODY = r"""
void evaluate(ivec2 pos,out D o[8]) {
    D a=ld(0,pos,0), b=ld(1,pos,0);
    o[0]=minD(a,b);
    D width=maxD(absD(U_edgeWidth),c(0.0001));
    D mask=smoothD(neg(width),width,sub(b,a));
    o[1]=mask; o[2]=mask; o[3]=mask; o[4]=c(1.0);
}
"""
PROGRAM = ShaderProgram("masked_union", BODY)
