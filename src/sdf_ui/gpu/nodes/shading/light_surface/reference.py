from ..light.reference import _height_gradient, positive_power

_MIN_LIGHT_LENGTH_SQUARED = 1e-8
_MIN_SHININESS = 1e-4


def _surface_diffuse_and_highlight(height, params):
    import numpy as np

    from sdf_ui.learning.autodiff import maximum, where

    gx, gy = _height_gradient(height)
    norm = (1.0 + gx * gx + gy * gy).sqrt()
    light = params["lightDir"] * np.array((1.0, -1.0, 1.0))
    light /= maximum((light * light).sum(), _MIN_LIGHT_LENGTH_SQUARED).sqrt()
    ndotl = (-gx * light[0] - gy * light[1] + light[2]) / norm
    lambert = maximum(ndotl, 0.0)
    highlight = positive_power(
        maximum(2.0 * ndotl / norm - light[2], 0.0),
        maximum(params["shininess"], _MIN_SHININESS),
    )
    return lambert, where(
        (lambert.data > 0.0) & (params["specular"].data > 0.0), highlight, 0.0
    )


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import maximum
    from sdf_ui.learning.render import _clip, _rgba, _tensor

    p = {name: _tensor(value) for name, value in params.items()}
    lambert, highlight = _surface_diffuse_and_highlight(inputs["heightField"], p)
    illumination = maximum(p["ambient"], 0.0) + maximum(p["diffuse"], 0.0) * lambert
    specular = maximum(p["specular"], 0.0) * highlight
    base = p["fgColor"]
    rgb = _clip(base[..., :3] * illumination[..., None] + specular[..., None], 0.0, 1.0)
    alpha = _clip(0.5 - inputs["sdf"], 0.0, 1.0) * base[..., 3]
    return _rgba(rgb, alpha)
