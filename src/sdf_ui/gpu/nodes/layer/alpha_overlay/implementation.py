"""Alpha-overlay extension's image interface and differentiable implementation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)


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
    "overlay", "gpu/nodes/layer/alpha_overlay/overlay.wgsl",
    (ShaderArgument("tex0", (4,), "rgba8"), ShaderArgument("tex1", (4,), "rgba8")),
    (), (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
