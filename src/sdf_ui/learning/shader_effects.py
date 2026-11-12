"""Executable float rules for standalone layer and postprocessing shaders.

``RULES[descriptor.name](context, inputs, params)`` returns a Tensor. Context
provides integer pixel grids ``x``, ``y`` as Tensors and ``size=(width, height)``;
inputs are keyed by GLSL image uniform name. Values are raw shader channels:
only to_rgb/to_lab convert colors, and rgba8 store clipping/quantization is left
to the caller. This module implements no plugin host render functions.

Blur follows the shader's asymmetric, single reflection exactly: the negative
tap uses abs(position - trunc(offset)); the positive tap changes sign when
position + loop_index >= image_extent - 1. The sign stays negative thereafter.
Remaining out-of-bounds imageLoad coordinates return zero in this CPU reference.
Dispatch metadata must advertise that border contract; GPU parity at such
coordinates requires the GPU to provide zero loads as well. Offsets truncate
toward zero and have connected zero derivatives away from integer jumps.
Dithering has identity derivatives away from thresholds; dither_1bit has true
connected zero derivatives for every input channel, including constant alpha=1.
"""

from functools import partial

from ..core.plugins.postprocessing.blur.implementation import DEFAULTS, reference as _blur
from ..core.plugins.layer.alpha_overlay.implementation import reference as _overlay
from ..core.plugins.layer.layer_mask.implementation import reference as _layer_mask
from ..core.plugins.layer.multiply.implementation import reference as _multiply
from ..core.plugins.layer.transparency.implementation import reference as _transparency
from ..core.plugins.postprocessing.invert.implementation import reference as _invert
from ..core.plugins.postprocessing.to_rgb.implementation import reference as _to_rgb
from ..core.plugins.postprocessing.to_lab.implementation import reference as _to_lab
from ..core.plugins.postprocessing.dithering.implementation import reference as _dithering
from ..core.plugins.postprocessing.dither_1bit.implementation import reference as _dither_1bit

__all__ = ["RULES", "DEFAULTS"]


RULES = {
    "overlay": _overlay,
    "layer_mask": _layer_mask,
    "multiply": _multiply,
    "transparency": _transparency,
    "invert": _invert,
    "to_rgb": _to_rgb,
    "to_lab": _to_lab,
    "dithering": _dithering,
    "dither_1bit": _dither_1bit,
}
RULES.update(
    {
        name: partial(_blur, name=name, axis=1 if "hor" in name else 0)
        for name in DEFAULTS
    }
)
