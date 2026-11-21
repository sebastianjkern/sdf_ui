"""Layer-mask extension's image interface and differentiable implementation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


def _alpha_composite(first, second, alpha):
    from sdf_ui.learning.autodiff import maximum, stack, where

    top_alpha = first[..., 3:4] * alpha
    out_alpha = top_alpha + second[..., 3:4] * (1.0 - top_alpha)
    premultiplied = first[..., :3] * top_alpha + second[..., :3] * second[
        ..., 3:4
    ] * (1.0 - top_alpha)
    rgb = where(out_alpha > 0, premultiplied / maximum(out_alpha, 1e-8), 0.0)
    return stack((rgb[..., 0], rgb[..., 1], rgb[..., 2], out_alpha[..., 0]), axis=-1)

def reference(context, inputs, params):
    import numpy as np

    from sdf_ui.learning.render import _tensor

    first, second = _tensor(inputs["tex0"]), _tensor(inputs["tex1"])
    alpha = _tensor(inputs["mask"])[..., 0:1]
    mode = params.get("mode", 0.0)
    mode = float(np.asarray(getattr(mode, "data", mode)))
    if mode < 0.5:
        return second * (1.0 - alpha) + first * alpha
    if mode < 3 / 2:
        blended = first * second
    elif mode < 5 / 2:
        blended = 1.0 - (1.0 - first) * (1.0 - second)
    elif mode < 7 / 2:
        return _alpha_composite(first, second, alpha)
    elif mode < 9 / 2:
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
    "gpu/nodes/layer/layer_mask/layer_mask.wgsl",
    tuple(ShaderArgument(name, (4,), "rgba8") for name in ("tex0", "tex1", "mask")),
    (ShaderArgument("mode", (), "float"),),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    zero_gradient_parameters=("mode",),
    behavior="smooth_for_fixed_sampling",
    notes="Piecewise derivatives; min/max ties split adjoints equally.",
    reference=reference,
    defaults={"mode": 0.0},
)
