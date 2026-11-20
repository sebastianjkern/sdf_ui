"""Blur shader layouts, defaults, stencil reference and GPU programs."""



from functools import partial

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)
from sdf_ui.gpu.nodes.layer.glsl import color_program
from sdf_ui.gpu.nodes.shader_expression import ShaderProgram

_OFFSET_TEXT = 'offset'

_WEIGHT_TEXT = 'weight'

WEIGHT_9 = (0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162)
WEIGHT_13 = (0.2094726562, 0.1832885742, 0.1221923828, 0.0610961914,
             0.0222167969, 0.0055541992, 0.0008544922, 0.0000610352)
DEFAULTS = {
    f"blur_{direction}_{width}": {
        _OFFSET_TEXT: tuple(float(i) for i in range(len(weights))), _WEIGHT_TEXT: weights,
    }
    for direction in ("hor", "ver")
    for width, weights in ((9, WEIGHT_9), (13, WEIGHT_13))
}


def _load_zero(image, x, y):
    import numpy as np

    from sdf_ui.learning.autodiff import where

    height, width = image.shape[:2]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    loaded = image[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]
    return where(valid[..., None], loaded, 0.0)



def _filter_stencil(image, x, y, position, extent, offset, weight, axis):
    import numpy as np

    count = len(weight)
    integer_offsets = np.trunc(offset.data).astype(np.intp)
    result = _load_zero(image, x, y) * weight[0] + offset.sum() * 0.0
    factor = np.ones_like(position)
    for index in range(1, count):
        delta = integer_offsets[index]
        negative = np.abs(position - delta)
        factor = np.where(position + index >= extent - 1, -1, factor)
        positive = position + factor * delta
        low_x, low_y = (negative, np.abs(y)) if axis == 1 else (np.abs(x), negative)
        high_x, high_y = (positive, y) if axis == 1 else (x, positive)
        samples = _load_zero(image, low_x, low_y) + _load_zero(image, high_x, high_y)
        result = result + samples * weight[index]
    return result

def _blur_axis(image, context, params, name, axis):
    import numpy as np

    from sdf_ui.learning.render import _tensor

    defaults = DEFAULTS[name]
    offset = _tensor(params.get(_OFFSET_TEXT, defaults[_OFFSET_TEXT]))
    weight = _tensor(params.get(_WEIGHT_TEXT, defaults[_WEIGHT_TEXT]))
    count = len(defaults[_WEIGHT_TEXT])
    if offset.shape != (count,) or weight.shape != (count,):
        raise ValueError("{} requires {} offsets and weights".format(name, count))
    x = np.trunc(context.x.data).astype(np.intp)
    y = np.trunc(context.y.data).astype(np.intp)
    position = x if axis == 1 else y
    return _filter_stencil(image, x, y, position, image.shape[axis], offset, weight, axis)


def reference(context, inputs, params, *, name, axis):
    from sdf_ui.learning.render import _tensor

    image = _tensor(inputs["origTex"])
    return _blur_axis(image, context, params, name, axis)


def gpu_body(component, count):
    delta = (
        "ivec2(int(U_offset(i).v),0)" if component == "x"
        else "ivec2(0,int(U_offset(i).v))"
    )
    return color_program("""
    D4 col=mul4(load4(0,pos),U_weight(0));
    int factor=1;
    for(int i=1;i<%d;i++) {
        ivec2 delta=%s;
        col=add4(col,mul4(load4(0,abs(pos-delta)),U_weight(i)));
        if(pos.%s+i>=size.%s-1) factor=-1;
        col=add4(col,mul4(load4(0,pos+factor*delta),U_weight(i)));
    }
""" % (count, delta, component, component))


def shader_parts(name):
    _, direction, width = name.split("_")
    count = len(DEFAULTS[name][_WEIGHT_TEXT])
    suffix = "hor" if direction == "hor" else "vert"
    interface = ShaderInterface(
        ImageBinding("destTex", 0, write=True), (ImageBinding("origTex", 1, read=True),),
        (Uniform(_OFFSET_TEXT, f"float[{count}]"), Uniform(_WEIGHT_TEXT, f"float[{count}]")),
    )
    contract = ShaderDifferentiation(
        name, f"gpu/nodes/postprocessing/blur/blur{width}_{suffix}.glsl",
        (ShaderArgument("origTex", (4,), "rgba8"),),
        (ShaderArgument(_OFFSET_TEXT, (count,), "float"), ShaderArgument(_WEIGHT_TEXT, (count,), "float")),
        (ShaderArgument("destTex", (4,), "rgba8"),),
        behavior="smooth_for_fixed_sampling", zero_gradient_parameters=(_OFFSET_TEXT,),
        notes="Transpose the actual reflected stencil; weights differentiate, truncated offsets have zero gradients.",
        sampling="asymmetric_reflected_zero_outside",
        reference=partial(reference, name=name, axis=1 if direction == "hor" else 0),
        defaults=DEFAULTS[name],
    )
    return interface, contract, ShaderProgram(name, gpu_body("x" if direction == "hor" else "y", count))
