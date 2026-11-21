"""Transform shader's image layout, sampling and differentiable rules."""



from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)

_FLOAT_TEXT = 'float'

def _bilinear(source, x, y):
    import numpy as np

    from sdf_ui.learning.render import _clip, _mix

    height, width = source.shape
    x, y = _clip(x, 0.0, width - 1), _clip(y, 0.0, height - 1)
    ix, iy = np.floor(x.data).astype(np.int64), np.floor(y.data).astype(np.int64)
    jx, jy = np.minimum(ix + 1, width - 1), np.minimum(iy + 1, height - 1)
    tx, ty = x - ix, y - iy
    upper = _mix(source[iy, ix], source[iy, jx], tx)
    lower = _mix(source[jy, ix], source[jy, jx], tx)
    return _mix(upper, lower, ty)


def _scale_coordinates(x, y, params):
    from sdf_ui.learning.render import _tensor

    factor = _tensor(params.get("factor", 0.0))
    if factor.item() == 0.0:
        raise ValueError("scale factor must be non-zero")
    return x / factor, y / factor, factor.abs()


def _rotate_coordinates(x, y, params):
    from sdf_ui.learning.render import _tensor

    angle = _tensor(params.get("angle", 0.0))
    return angle.cos() * x + angle.sin() * y, -angle.sin() * x + angle.cos() * y, 1.0


def _skew_coordinates(x, y, params, mode):
    from sdf_ui.learning.autodiff import maximum
    from sdf_ui.learning.render import _length, _tensor

    skew = _tensor(params.get("skew", (0.0, 0.0)))
    determinant = 1.0 - skew[0] * skew[1]
    if determinant.item() == 0.0:
        raise ValueError("skew values produce a singular transform")
    x, y = (x - skew[0] * y) / determinant, (y - skew[1] * x) / determinant
    scale = 1.0
    if mode > 5 / 2:
        scale = maximum(
            0.5 * (
                _length(1.0 / determinant, -skew[1] / determinant)
                + _length(-skew[0] / determinant, 1.0 / determinant)
            ),
            1e-6,
        )
    return x, y, scale


def _centered_coordinates(context, params, mode):
    from sdf_ui.learning.render import _tensor

    center = _tensor(params.get("center", (0.0, 0.0)))
    x, y = context.x - center[0], context.y - center[1]
    if mode < 3 / 2:
        x, y, scale = _scale_coordinates(x, y, params)
    elif mode < 5 / 2:
        x, y, scale = _rotate_coordinates(x, y, params)
    else:
        x, y, scale = _skew_coordinates(x, y, params, mode)
    return x + center[0], y + center[1], scale


def reference(context, inputs, params):
    from sdf_ui.learning.render import _tensor

    kind = _tensor(params["transform_kind"])
    mode = kind.item()
    if mode < 0.5:
        offset = _tensor(params.get("offset", (0.0, 0.0)))
        x, y, scale = context.x - offset[0], context.y - offset[1], 1.0
    else:
        x, y, scale = _centered_coordinates(context, params, mode)
    return _bilinear(inputs["sdf"], x, y) * scale + kind * 0.0


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("sdf", 1, read=True),),
    (Uniform("transform_kind", _FLOAT_TEXT), Uniform("offset", "vec2"),
     Uniform("center", "vec2"), Uniform("factor", _FLOAT_TEXT),
     Uniform("angle", _FLOAT_TEXT), Uniform("skew", "vec2")),
)
CONTRACT = ShaderDifferentiation(
    "transform", "gpu/nodes/primitives/transform/transform.wgsl",
    (ShaderArgument("sdf", (), "r32f"),),
    (ShaderArgument("transform_kind", (), _FLOAT_TEXT), ShaderArgument("offset", (2,), "vec2"),
     ShaderArgument("center", (2,), "vec2"), ShaderArgument("factor", (), _FLOAT_TEXT),
     ShaderArgument("angle", (), _FLOAT_TEXT), ShaderArgument("skew", (2,), "vec2")),
    (ShaderArgument("destTex", (), "r32f"),), zero_gradient_parameters=("transform_kind",),
    notes="Bilinear source gradients scatter-add; transform_kind is a discrete mode.",
    sampling="clamped_bilinear", reference=reference,
)
