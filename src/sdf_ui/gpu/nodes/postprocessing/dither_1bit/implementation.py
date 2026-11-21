"""One-bit dithering's shader interface and true zero-gradient rule."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)

from ..bayer import threshold


def reference(context, inputs, params):
    import numpy as np

    from sdf_ui.learning.render import _rgba, _tensor

    color = _tensor(inputs["origTex"])
    luminance = np.sum(color.data[..., :3] * (0.299, 0.587, 0.114), axis=-1)
    step = (luminance >= threshold(context)).astype(float)
    return _rgba(color[..., :3] * 0.0 + step[..., None], color[..., 3] * 0.0 + 1.0)


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("origTex", 1, read=True),)
)
CONTRACT = ShaderDifferentiation(
    "dither_1bit", "gpu/nodes/postprocessing/dither_1bit/dither_1bit.wgsl",
    (ShaderArgument("origTex", (4,), "rgba8"),), (),
    (ShaderArgument("destTex", (4,), "rgba8"),), behavior="zero_almost_everywhere",
    notes="True zero derivative almost everywhere; undefined at thresholds. No straight-through estimator.",
    reference=reference,
)
