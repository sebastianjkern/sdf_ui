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

import numpy as np

from .autodiff import where

__all__ = ["RULES", "DEFAULTS"]


def _tensor(value):
    # Lazy imports let render install these rules without a circular import.
    from .render import _tensor

    return _tensor(value)


def _overlay(context, inputs, params):
    from .render import _rgba

    top, base = _tensor(inputs["tex0"]), _tensor(inputs["tex1"])
    a, b = top[..., 3], base[..., 3]
    alpha = a + (1.0 - a) * b
    positive = alpha.data > 0.0
    denominator = where(positive, alpha, 1.0)[..., None]
    rgb = (
        top[..., :3] * a[..., None] + base[..., :3] * ((1.0 - a) * b)[..., None]
    ) / denominator
    return _rgba(where(positive[..., None], rgb, 0.0), alpha)


def _layer_mask(context, inputs, params):
    first, second = _tensor(inputs["tex0"]), _tensor(inputs["tex1"])
    alpha = _tensor(inputs["mask"])[..., 0:1]
    return second * (1.0 - alpha) + first * alpha


def _multiply(context, inputs, params):
    return _tensor(inputs["mask1"]) * _tensor(inputs["mask2"])


def _transparency(context, inputs, params):
    from .render import _rgba

    color = _tensor(inputs["tex0"])
    return _rgba(color[..., :3], color[..., 3] * _tensor(params["alpha"]))


def _invert(context, inputs, params):
    from .render import _rgba

    color = _tensor(inputs["origTex"])
    return _rgba(1.0 - color[..., :3], color[..., 3])


def _to_rgb(context, inputs, params):
    from .render import _lab_to_rgb

    return _lab_to_rgb(_tensor(inputs["origTex"]))


def _to_lab(context, inputs, params):
    from .render import _rgb_to_lab

    return _rgb_to_lab(_tensor(inputs["origTex"]))


_BAYER = np.array(
    (
        (0, 32, 8, 40, 2, 34, 10, 42),
        (48, 16, 56, 24, 50, 18, 58, 26),
        (12, 44, 4, 36, 14, 46, 6, 38),
        (60, 28, 52, 20, 62, 30, 54, 22),
        (3, 35, 11, 43, 1, 33, 9, 41),
        (51, 19, 59, 27, 49, 17, 57, 25),
        (15, 47, 7, 39, 13, 45, 5, 37),
        (63, 31, 55, 23, 61, 29, 53, 21),
    )
)


def _threshold(context):
    x = np.trunc(context.x.data).astype(np.intp) % 8
    y = np.trunc(context.y.data).astype(np.intp) % 8
    # GLSL indexes dither[x][y], rather than the usual image-row convention.
    return (_BAYER[x, y] + 1.0) / 64.0


def _dithering(context, inputs, params):
    from .render import _rgba

    color = _tensor(inputs["origTex"])
    step = (color.data[..., :3] >= _threshold(context)[..., None]) / 255.0
    return _rgba(color[..., :3] + step, color[..., 3])


def _dither_1bit(context, inputs, params):
    from .render import _rgba

    color = _tensor(inputs["origTex"])
    lum = np.sum(color.data[..., :3] * (0.299, 0.587, 0.114), axis=-1)
    step = (lum >= _threshold(context)).astype(float)
    return _rgba(color[..., :3] * 0.0 + step[..., None], color[..., 3] * 0.0 + 1.0)


_WEIGHT_9 = (0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162)
_WEIGHT_13 = (
    0.2094726562,
    0.1832885742,
    0.1221923828,
    0.0610961914,
    0.0222167969,
    0.0055541992,
    0.0008544922,
    0.0000610352,
)

DEFAULTS = {
    "blur_{}_{}".format(direction, width): {
        "offset": tuple(float(i) for i in range(len(weights))),
        "weight": weights,
    }
    for direction in ("hor", "ver")
    for width, weights in ((9, _WEIGHT_9), (13, _WEIGHT_13))
}


def _load_zero(image, x, y):
    """Gather with zero OOB values and scatter-add input gradients."""
    height, width = image.shape[:2]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    loaded = image[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]
    return where(valid[..., None], loaded, 0.0)


def _blur(context, inputs, params, *, name, axis):
    image = _tensor(inputs["origTex"])
    defaults = DEFAULTS[name]
    offset = _tensor(params.get("offset", defaults["offset"]))
    weight = _tensor(params.get("weight", defaults["weight"]))
    count = len(defaults["weight"])
    if offset.shape != (count,) or weight.shape != (count,):
        raise ValueError("{} requires {} offsets and weights".format(name, count))
    x = np.trunc(context.x.data).astype(np.intp)
    y = np.trunc(context.y.data).astype(np.intp)
    position = x if axis == 1 else y
    extent = image.shape[axis]
    integer_offsets = np.trunc(offset.data).astype(np.intp)
    # offset[0] is unused in GLSL, but still receives an explicit zero gradient.
    result = _load_zero(image, x, y) * weight[0] + offset.sum() * 0.0
    factor = np.ones_like(position)
    for i in range(1, count):
        delta = integer_offsets[i]
        negative = np.abs(position - delta)
        # The condition deliberately uses i, not the supplied offset value.
        factor = np.where(position + i >= extent - 1, -1, factor)
        positive = position + factor * delta
        if axis == 1:
            low = _load_zero(image, negative, np.abs(y))
            high = _load_zero(image, positive, y)
        else:
            low = _load_zero(image, np.abs(x), negative)
            high = _load_zero(image, x, positive)
        result = result + (low + high) * weight[i]
    return result


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
