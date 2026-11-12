"""Alpha-overlay extension's image interface and differentiable implementation."""

from sdf_ui.core.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.core.plugins.base import ImageBinding, ShaderInterface
from sdf_ui.core.shader_expression import ShaderProgram
from ..glsl import color_program


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import where
    from sdf_ui.learning.render import _rgba, _tensor

    top, base = _tensor(inputs["tex0"]), _tensor(inputs["tex1"])
    a, b = top[..., 3], base[..., 3]
    alpha = a + (1.0 - a) * b
    positive = alpha.data > 0.0
    denominator = where(positive, alpha, 1.0)[..., None]
    rgb = (top[..., :3] * a[..., None] + base[..., :3] * ((1.0 - a) * b)[..., None]) / denominator
    return _rgba(where(positive[..., None], rgb, 0.0), alpha)


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    (ImageBinding("tex0", 1, read=True), ImageBinding("tex1", 2, read=True)),
)
CONTRACT = ShaderDifferentiation(
    "overlay", "core/plugins/layer/alpha_overlay/shader.glsl",
    (ShaderArgument("tex0", (4,), "rgba8"), ShaderArgument("tex1", (4,), "rgba8")),
    (), (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
BODY = color_program(r"""
    D4 top=load4(0,pos), base=load4(1,pos);
    D remaining=sub(c(1.0),top.w);
    D alpha=add(top.w,mul(remaining,base.w));
    D3 rgb=v3(vec3(0.0));
    if(alpha.v>0.0) {
        D3 topRgb=D3(top.x,top.y,top.z);
        D3 baseRgb=D3(base.x,base.y,base.z);
        rgb=div3(add3(mul3(topRgb,top.w),
                      mul3(mul3(baseRgb,remaining),base.w)),alpha);
    }
    D4 col=D4(rgb.x,rgb.y,rgb.z,alpha);
""")
PROGRAM = ShaderProgram("overlay", BODY)
