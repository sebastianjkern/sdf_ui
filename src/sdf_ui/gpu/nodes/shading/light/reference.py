"""Light shader's differentiable height, normal and material evaluation."""

from ..sampling import sample

_MIN_HEIGHT = 0.0001
_PROFILE_STEP = 1.5
_PI = 3.1415926538
_PROFILE_THRESHOLD_2 = 2.5
_PROFILE_THRESHOLD_3 = 3.5
_PROFILE_THRESHOLD_4 = 4.5
_SOBEL_SCALE = 0.125
_HEIGHT_BAND_WIDTH = 0.08


def positive_power(base, exponent):
    from sdf_ui.learning.autodiff import where

    positive = base.data > 0.0
    return where(positive, where(positive, base, 1.0) ** exponent, 0.0)


def height_curve(value, params):
    from sdf_ui.learning.autodiff import maximum
    from sdf_ui.learning.render import _clip, _smoothstep

    t = positive_power(_clip(value, 0.0, 1.0), maximum(params["heightGamma"], _MIN_HEIGHT))
    profile = params["heightProfile"].item()
    if profile < 0.5:
        return _smoothstep(0.0, 1.0, t)
    if profile < _PROFILE_STEP:
        return 0.5 - 0.5 * (t * _PI).cos()
    if profile < _PROFILE_THRESHOLD_2:
        return t * t
    if profile < _PROFILE_THRESHOLD_3:
        return 1.0 - (1.0 - t) * (1.0 - t)
    if profile < _PROFILE_THRESHOLD_4:
        return maximum(1.0 - (1.0 - t) * (1.0 - t), 0.0).sqrt()
    return 1.0 - maximum(1.0 - t * t, 0.0).sqrt()


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import maximum, where
    from sdf_ui.learning.render import _clip, _length, _mix, _rgba, _smoothstep, _tensor

    p = {name: _tensor(value) for name, value in params.items()}
    distance = inputs["sdf"] - p["inflate"]
    coverage = 1.0 - _smoothstep(p["inner"], p["outer"] + _MIN_HEIGHT, distance)
    height_map = _height_map(distance, coverage, p, maximum, where, _smoothstep)
    gx, gy = _height_gradient(height_map)
    return _shade_material(
        context, p, coverage, height_map, gx, gy,
        maximum, where, _clip, _length, _mix, _rgba, _smoothstep,
    )


def _height_map(distance, coverage, params, maximum, where, smoothstep):
    if params["shadeBackground"].item() < 0.5:
        return coverage
    bevel = maximum(params["bevel"], _MIN_HEIGHT)
    background_bevel = maximum(
        where(params["backgroundBevel"].data > 0.0, params["backgroundBevel"], bevel),
        _MIN_HEIGHT,
    )
    t = params["shadeBackground"] * (
        1.0 - smoothstep(0.0, background_bevel, maximum(distance, 0.0))
    )
    return height_curve(t, params)


def _height_gradient(height_map):
    h00, h10, h20 = sample(height_map, -1, -1), sample(height_map, 0, -1), sample(height_map, 1, -1)
    h01, h21 = sample(height_map, -1, 0), sample(height_map, 1, 0)
    h02, h12, h22 = sample(height_map, -1, 1), sample(height_map, 0, 1), sample(height_map, 1, 1)
    gx = ((h20 + 2.0 * h21 + h22) - (h00 + 2.0 * h01 + h02)) * _SOBEL_SCALE
    gy = ((h02 + 2.0 * h12 + h22) - (h00 + 2.0 * h10 + h20)) * _SOBEL_SCALE
    return gx, gy


def _shade_material(context, p, coverage, height_map, gx, gy, maximum, where,
                    clip, length, mix, rgba, smoothstep):
    sx = clip(-gx * p["normalStrength"], -8.0, 8.0)
    sy = clip(-gy * p["normalStrength"], -8.0, 8.0)
    shade, highlight = _light_response(context, p, sx, sy, maximum, where)
    base = mix(p["bgColor"], p["fgColor"], coverage[..., None])
    rgb = clip(
        base[..., :3] * shade[..., None] + (maximum(p["specular"], 0.0) * highlight)[..., None],
        0.0, 1.0,
    )
    lit = rgba(rgb, base[..., 3])
    slope = length(gx, gy) * p["normalStrength"]
    height_band = smoothstep(0.0, _HEIGHT_BAND_WIDTH, height_map) * smoothstep(
        0.0, _HEIGHT_BAND_WIDTH, 1.0 - height_map
    )
    influence = height_band * smoothstep(0.0, _HEIGHT_BAND_WIDTH, slope)
    return mix(base, lit, influence[..., None])


def _light_response(context, params, sx, sy, maximum, where):
    import numpy as np

    z = maximum(params["height"], _MIN_HEIGHT)
    norm = (sx * sx + sy * sy + z * z).sqrt()
    nx, ny, nz = sx / norm, sy / norm, z / norm
    light = params["lightDir"] * np.array((1.0, -1.0, 1.0))
    light = light / (light * light).sum().sqrt()
    dot = nx * light[0] + ny * light[1] + nz * light[2]
    lambert = maximum(dot, 0.0)
    reflected_z = maximum(2.0 * dot * nz - light[2], 0.0)
    highlight = positive_power(reflected_z, maximum(params["shininess"], _MIN_HEIGHT))
    highlight = where(
        (lambert.data > 0.0) & (params["specular"].data > 0.0), highlight, 0.0
    )
    shade = maximum(params["ambient"], 0.0) + maximum(params["diffuse"], 0.0) * lambert
    return shade, highlight
