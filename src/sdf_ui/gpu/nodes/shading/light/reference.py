"""Light shader's differentiable height, normal and material evaluation."""

from ..sampling import sample


def positive_power(base, exponent):
    from sdf_ui.learning.autodiff import where

    positive = base.data > 0.0
    return where(positive, where(positive, base, 1.0) ** exponent, 0.0)


def height_curve(value, params):
    from sdf_ui.learning.autodiff import maximum
    from sdf_ui.learning.render import _clip, _smoothstep

    t = positive_power(_clip(value, 0.0, 1.0), maximum(params["heightGamma"], 0.0001))
    profile = params["heightProfile"].item()
    if profile < 0.5:
        return _smoothstep(0.0, 1.0, t)
    if profile < 1.5:
        return 0.5 - 0.5 * (t * 3.1415926538).cos()
    if profile < 2.5:
        return t * t
    if profile < 3.5:
        return 1.0 - (1.0 - t) * (1.0 - t)
    if profile < 4.5:
        return maximum(1.0 - (1.0 - t) * (1.0 - t), 0.0).sqrt()
    return 1.0 - maximum(1.0 - t * t, 0.0).sqrt()


def reference(context, inputs, params):
    import numpy as np

    from sdf_ui.learning.autodiff import maximum, where
    from sdf_ui.learning.render import _clip, _length, _mix, _rgba, _smoothstep, _tensor

    p = {name: _tensor(value) for name, value in params.items()}
    distance = inputs["sdf"] - p["inflate"]
    coverage = 1.0 - _smoothstep(p["inner"], p["outer"] + 0.0001, distance)
    if p["shadeBackground"].item() < 0.5:
        height_map = coverage
    else:
        bevel = maximum(p["bevel"], 0.0001)
        bg_bevel = maximum(
            where(p["backgroundBevel"].data > 0.0, p["backgroundBevel"], bevel), 0.0001
        )
        t = p["shadeBackground"] * (
            1.0 - _smoothstep(0.0, bg_bevel, maximum(distance, 0.0))
        )
        height_map = height_curve(t, p)

    h00 = sample(height_map, -1, -1)
    h10 = sample(height_map, 0, -1)
    h20 = sample(height_map, 1, -1)
    h01 = sample(height_map, -1, 0)
    h21 = sample(height_map, 1, 0)
    h02 = sample(height_map, -1, 1)
    h12 = sample(height_map, 0, 1)
    h22 = sample(height_map, 1, 1)
    gx = ((h20 + 2.0 * h21 + h22) - (h00 + 2.0 * h01 + h02)) * 0.125
    gy = ((h02 + 2.0 * h12 + h22) - (h00 + 2.0 * h10 + h20)) * 0.125
    sx = _clip(-gx * p["normalStrength"], -8.0, 8.0)
    sy = _clip(-gy * p["normalStrength"], -8.0, 8.0)
    z = maximum(p["height"], 0.0001)
    norm = (sx * sx + sy * sy + z * z).sqrt()
    nx, ny, nz = sx / norm, sy / norm, z / norm
    light = p["lightDir"] * np.array((1.0, -1.0, 1.0))
    light = light / (light * light).sum().sqrt()
    dot = nx * light[0] + ny * light[1] + nz * light[2]
    lambert = maximum(dot, 0.0)
    reflected_z = maximum(2.0 * dot * nz - light[2], 0.0)
    highlight = positive_power(reflected_z, maximum(p["shininess"], 0.0001))
    highlight = where((lambert.data > 0.0) & (p["specular"].data > 0.0), highlight, 0.0)
    shade = maximum(p["ambient"], 0.0) + maximum(p["diffuse"], 0.0) * lambert
    base = _mix(p["bgColor"], p["fgColor"], coverage[..., None])
    rgb = _clip(
        base[..., :3] * shade[..., None] + (maximum(p["specular"], 0.0) * highlight)[..., None],
        0.0, 1.0,
    )
    lit = _rgba(rgb, base[..., 3])
    slope = _length(gx, gy) * p["normalStrength"]
    height_band = _smoothstep(0.0, 0.08, height_map) * _smoothstep(0.0, 0.08, 1.0 - height_map)
    influence = height_band * _smoothstep(0.0, 0.08, slope)
    return _mix(base, lit, influence[..., None])
