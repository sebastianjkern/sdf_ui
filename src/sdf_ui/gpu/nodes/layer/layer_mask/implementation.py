"""Layer-mask extension's image interface and differentiable implementation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)
from sdf_ui.gpu.nodes.shader_expression import ShaderProgram

from ..glsl import color_program


def reference(context, inputs, params):
    import numpy as np

    from sdf_ui.learning.render import _tensor

    first, second = _tensor(inputs["tex0"]), _tensor(inputs["tex1"])
    alpha = _tensor(inputs["mask"])[..., 0:1]
    mode = params.get("mode", 0.0)
    mode = float(np.asarray(getattr(mode, "data", mode)))
    if mode < 0.5:
        return second * (1.0 - alpha) + first * alpha
    if mode < 1.5:
        blended = first * second
    elif mode < 2.5:
        blended = 1.0 - (1.0 - first) * (1.0 - second)
    elif mode < 3.5:
        from sdf_ui.learning.autodiff import maximum, stack, where

        top_alpha = first[..., 3:4] * alpha
        out_alpha = top_alpha + second[..., 3:4] * (1.0 - top_alpha)
        premultiplied = first[..., :3] * top_alpha + second[..., :3] * second[
            ..., 3:4
        ] * (1.0 - top_alpha)
        rgb = premultiplied / maximum(out_alpha, 1e-8)

        rgb = where(out_alpha > 0, rgb, 0.0)
        return stack(
            (rgb[..., 0], rgb[..., 1], rgb[..., 2], out_alpha[..., 0]), axis=-1
        )
    elif mode < 4.5:
        from sdf_ui.learning.autodiff import minimum

        blended = minimum(first, second)
    else:
        from sdf_ui.learning.autodiff import maximum

        blended = maximum(first, second)
    return second * (1.0 - alpha) + blended * alpha


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    (
        ImageBinding("tex0", 1, read=True),
        ImageBinding("tex1", 2, read=True),
        ImageBinding("mask", 3, read=True),
    ),
    (Uniform("mode", "float"),),
)
CONTRACT = ShaderDifferentiation(
    "layer_mask",
    "gpu/nodes/layer/layer_mask/shader.glsl",
    tuple(ShaderArgument(name, (4,), "rgba8") for name in ("tex0", "tex1", "mask")),
    (ShaderArgument("mode", (), "float"),),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    zero_gradient_parameters=("mode",),
    behavior="smooth_for_fixed_sampling",
    notes="Piecewise derivatives; min/max ties split adjoints equally.",
    reference=reference,
    defaults={"mode": 0.0},
)
BODY = color_program(r"""
    D4 first=load4(0,pos), second=load4(1,pos);
    D alpha=ld(2,pos,0);
    D4 mixed=mix4(second,first,alpha);
    D4 product=D4(mul(first.x,second.x),mul(first.y,second.y),
                  mul(first.z,second.z),mul(first.w,second.w));
    D4 screen=D4(sub(c(1.0),mul(sub(c(1.0),first.x),sub(c(1.0),second.x))),
                 sub(c(1.0),mul(sub(c(1.0),first.y),sub(c(1.0),second.y))),
                 sub(c(1.0),mul(sub(c(1.0),first.z),sub(c(1.0),second.z))),
                 first.w);
    D4 darken=D4(minD(first.x,second.x),minD(first.y,second.y),
                 minD(first.z,second.z),minD(first.w,second.w));
    D4 lighten=D4(maxD(first.x,second.x),maxD(first.y,second.y),
                  maxD(first.z,second.z),maxD(first.w,second.w));
    D4 col=mixed;
    if(U_mode.v>0.5 && U_mode.v<1.5) col=mix4(second,product,alpha);
    else if(U_mode.v>1.5 && U_mode.v<2.5) col=mix4(second,screen,alpha);
    else if(U_mode.v>2.5 && U_mode.v<3.5) {
        D topAlpha=mul(first.w,alpha);
        D remaining=sub(c(1.0),topAlpha);
        D outAlpha=add(topAlpha,mul(second.w,remaining));
        D3 rgb=D3(vec3(0.0));
        if(outAlpha.v>0.0) {
            D3 top=D3(first.x,first.y,first.z);
            D3 base=D3(second.x,second.y,second.z);
            rgb=div3(add3(mul3(top,topAlpha),
                          mul3(mul3(base,second.w),remaining)),outAlpha);
        }
        col=D4(rgb.x,rgb.y,rgb.z,outAlpha);
    }
    else if(U_mode.v>3.5 && U_mode.v<4.5) col=mix4(second,darken,alpha);
    else if(U_mode.v>4.5) col=mix4(second,lighten,alpha);
""")
PROGRAM = ShaderProgram("layer_mask", BODY)
