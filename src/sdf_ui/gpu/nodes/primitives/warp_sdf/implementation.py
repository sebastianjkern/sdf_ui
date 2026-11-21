"""Differentiable reference and shader contract for vector-field warping."""



import numpy as np

from sdf_ui.gpu.nodes.differentiation import ShaderArgument, ShaderDifferentiation

_STEPS_TEXT = 'steps'

def _sample(image, x, y):
    from sdf_ui.learning.autodiff import where

    height, width = image.shape[:2]
    x0 = np.floor(x.data).astype(np.int64)
    y0 = np.floor(y.data).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = x - x0, y - y0
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)
    top = image[y0c, x0c] * (1 - fx) + image[y0c, x1c] * fx
    bottom = image[y1c, x0c] * (1 - fx) + image[y1c, x1c] * fx
    inside = (x.data >= 0) & (y.data >= 0) & (x.data <= width - 1) & (y.data <= height - 1)
    return where(inside, top * (1 - fy) + bottom * fy, 0.0)


def _reference(context, inputs, params, color=False):
    from sdf_ui.learning.autodiff import stack
    from sdf_ui.learning.render import _tensor

    field = inputs["fieldTex"]
    x, y = context.x, context.y
    strength = _tensor(params["strength"])
    steps = int(params[_STEPS_TEXT].item())
    for _ in range(steps):
        vx = _sample(field[..., 0], x, y)
        vy = _sample(field[..., 1], x, y)
        x = x - vx * strength / steps
        y = y - vy * strength / steps
    source = inputs["sourceTex"]
    if color:
        channels = tuple(_sample(source[..., index], x, y) for index in range(4))
        return stack(channels, axis=-1)
    sampled = _sample(source, x, y)
    inside = (
        (x.data >= 0)
        & (y.data >= 0)
        & (x.data <= context.x.shape[-1] - 1)
        & (y.data <= context.y.shape[-2] - 1)
    )
    from sdf_ui.learning.autodiff import where

    return where(inside, sampled, float(max(context.x.shape[-1], context.y.shape[-2])))


SDF_CONTRACT = ShaderDifferentiation(
    "warp_sdf",
    "gpu/nodes/primitives/warp_sdf/warp_sdf.wgsl",
    (ShaderArgument("sourceTex", (), "r32f"), ShaderArgument("fieldTex", (4,), "rgba32float")),
    (ShaderArgument("strength", (), "float"), ShaderArgument(_STEPS_TEXT, (), "int")),
    (ShaderArgument("destTex", (), "r32f"),),
    zero_gradient_parameters=(_STEPS_TEXT,), sampling="linear_clamp",
    notes="Differentiate bilinear weights within a fixed cell; integration steps are discrete.",
    reference=lambda context, inputs, params: _reference(context, inputs, params),
)
COLOR_CONTRACT = ShaderDifferentiation(
    "warp_color",
    "gpu/nodes/shading/warp_color/warp_color.wgsl",
    (ShaderArgument("sourceTex", (4,), "rgba8"), ShaderArgument("fieldTex", (4,), "rgba32float")),
    (ShaderArgument("strength", (), "float"), ShaderArgument(_STEPS_TEXT, (), "int")),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    zero_gradient_parameters=(_STEPS_TEXT,), sampling="linear_clamp",
    notes="Differentiate bilinear weights within a fixed cell; integration steps are discrete.",
    reference=lambda context, inputs, params: _reference(context, inputs, params, color=True),
)
