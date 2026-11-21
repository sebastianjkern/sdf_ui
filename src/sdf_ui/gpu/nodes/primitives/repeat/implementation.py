"""Repeat shader's integer sampling and differentiable implementation."""



from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)

from ..segment_math import minimum_candidates

_REPEAT_TEXT = 'repeat'

def reference(context, inputs, params):
    import numpy as np

    from sdf_ui.learning.autodiff import Tensor, where
    from sdf_ui.learning.render import _tensor

    def load_zero(source, x, y):
        ix, iy = np.trunc(x).astype(np.int64), np.trunc(y).astype(np.int64)
        height, width = source.shape
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        return where(valid, source[np.clip(iy, 0, height - 1), np.clip(ix, 0, width - 1)], 0.0)

    spacing = _tensor(params[_REPEAT_TEXT])
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
    (Uniform(_REPEAT_TEXT, "float"),),
)
CONTRACT = ShaderDifferentiation(
    _REPEAT_TEXT, "gpu/nodes/primitives/repeat/repeat.wgsl",
    (ShaderArgument("sdf0", (), "r32f"),), (ShaderArgument(_REPEAT_TEXT, (), "float"),),
    (ShaderArgument("destTex", (), "r32f"),),
    zero_gradient_parameters=(_REPEAT_TEXT,), sampling="integer_zero_outside",
    notes="Integer imageLoad addresses: spacing derivative zero away from jumps, source gradients accumulate.",
    reference=reference,
)
