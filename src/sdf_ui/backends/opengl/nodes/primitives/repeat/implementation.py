"""Repeat shader's integer sampling and differentiable implementation."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram
from ..segment_math import minimum_candidates


def reference(context, inputs, params):
    import numpy as np
    from sdf_ui.learning.autodiff import Tensor, where
    from sdf_ui.learning.render import _tensor

    def load_zero(source, x, y):
        ix, iy = np.trunc(x).astype(np.int64), np.trunc(y).astype(np.int64)
        height, width = source.shape
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        return where(valid, source[np.clip(iy, 0, height - 1), np.clip(ix, 0, width - 1)], 0.0)

    spacing = _tensor(params["repeat"])
    if spacing.item() == 0.0:
        raise ValueError("repeat spacing must be non-zero")
    x, y, s = context.x.data, context.y.data, spacing.item()
    ix, iy = np.rint(x / s), np.rint(y / s)
    ox, oy = np.sign(x - s * ix), np.sign(y - s * iy)
    candidates = [Tensor(np.full(context.x.shape, 1e20))]
    for j in range(2):
        for i in range(2):
            candidates.append(
                load_zero(inputs["sdf0"], x - s * (ix + i * ox), y - s * (iy + j * oy))
            )
    return minimum_candidates(*candidates) + spacing * 0.0


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("sdf0", 1, read=True),),
    (Uniform("repeat", "float"),),
)
CONTRACT = ShaderDifferentiation(
    "repeat", "backends/opengl/nodes/primitives/repeat/shader.glsl",
    (ShaderArgument("sdf0", (), "r32f"),), (ShaderArgument("repeat", (), "float"),),
    (ShaderArgument("destTex", (), "r32f"),),
    zero_gradient_parameters=("repeat",), sampling="integer_zero_outside",
    notes="Integer imageLoad addresses: spacing derivative zero away from jumps, source gradients accumulate.",
    reference=reference,
)
BODY = r"""
void evaluate(ivec2 pos,out D o[8]) {
    vec2 p=vec2(pos);
    // Integer addressing is discrete, including all dependence on spacing.
    float spacing=U_repeat.v;
    vec2 cell=roundEven(p/spacing), direction=sign(p-spacing*cell);
    D candidates[5]; candidates[0]=c(1e20);
    for(int j=0;j<2;j++) for(int i=0;i<2;i++) {
        vec2 address=p-spacing*(cell+vec2(i,j)*direction);
        candidates[1+j*2+i]=ld(0,ivec2(address),0);
    }
    float value=candidates[0].v, derivative=0.0, count=0.0;
    for(int i=1;i<5;i++) value=min(value,candidates[i].v);
    for(int i=0;i<5;i++) if(candidates[i].v==value) {
        derivative+=candidates[i].d; count+=1.0;
    }
    o[0]=D(value,derivative/count);
}
"""
PROGRAM = ShaderProgram("repeat", BODY)
