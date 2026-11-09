"""Executable float reference math for standalone shading shader descriptors.

``RULES[name](context, inputs, params)`` consumes GLSL image/uniform names.
The context supplies Tensor pixel grids ``x``, ``y`` and ``size=(width, height)``.
Results are unquantized (height, width, 4) Tensors, with row zero at the bottom.
RGB/LAB conversion follows each shader, rather than imposing a common space.
Continuous expressions retain their autodiff graphs; branch decisions and
modulo cell indices use NumPy booleans/integers and piecewise derivatives.
Noise uses fixed float32 pixel coordinates and has no trainable uniforms.
"""

import numpy as np

from .autodiff import Tensor, maximum, minimum, stack, where
from .render import _clip, _length, _mix, _rgb_to_lab, _rgba, _smoothstep, _tensor

__all__ = ["RULES"]


def _clear_color(context, inputs, params):
    return _rgb_to_lab(_tensor(params["color"])) + (context.x * 0.0)[..., None]


def _fill(context, inputs, params):
    distance = inputs["sdf"] - _tensor(params["inflate"])
    weight = _smoothstep(
        _tensor(params["first"]), _tensor(params["second"]) + 0.0001, distance
    )
    return _mix(
        _rgb_to_lab(_tensor(params["color"])),
        _rgb_to_lab(_tensor(params["background"])),
        weight[..., None],
    )


def _outline(context, inputs, params):
    distance = abs(inputs["sdf"] - _tensor(params["inflate"]))
    # OUTLINE_DISTANCE is the integer macro -1: both divisions by 2 in
    # the shader truncate to zero before AA_DISTANCE is added.
    weight = _smoothstep(0.0, 1.5001, distance)
    return _mix(
        _rgb_to_lab(_tensor(params["outline"])),
        _rgb_to_lab(_tensor(params["background"])),
        weight[..., None],
    )


def _fill_from_texture(context, inputs, params):
    distance = inputs["sdf"] - _tensor(params["inflate"])
    weight = _smoothstep(0.0, 1.5001, distance)
    return _mix(inputs["origTex"], _tensor(params["background"]), weight[..., None])


def _isolines(context, inputs, params):
    distance = abs(inputs["sdf"] - _tensor(params["inflate"]))
    # The engine rejects nonfinite Tensor values. Mask the shader's remaining
    # invalid-distance case before modulo so the inactive path stays finite.
    valid = distance.data <= 1e20
    distance = where(valid, distance, 0.0)
    period = maximum(_tensor(params["spacing"]), 1e-6)
    shifted = distance + _tensor(params["phase"])
    wrapped = shifted - np.floor((shifted / period).data) * period
    wrapped = where(wrapped.data < 0.0, wrapped + period, wrapped)
    band = minimum(wrapped, period - wrapped)
    half_width = maximum(_tensor(params["line_width"]) * 0.5, 0.0)
    aa = 1.5001 + maximum(_tensor(params["feather"]), 0.0)
    line = 1.0 - _smoothstep(maximum(half_width - aa, 0.0), half_width + aa, band)
    line = where(valid, line, 0.0)
    return _mix(
        _rgb_to_lab(_tensor(params["background"])),
        _rgb_to_lab(_tensor(params["line_color"])),
        line[..., None],
    )


def _sample(value, dx, dy, *, zero=False):
    """Integer imageLoad with either explicit OOB zero or clamped coordinates."""
    height, width = value.shape[:2]
    y, x = np.indices((height, width))
    x, y = x + dx, y + dy
    result = value[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]
    if zero:
        result = where((x >= 0) & (x < width) & (y >= 0) & (y < height), result, 0.0)
    return result


def _partial_derivative(context, inputs, params):
    sdf = inputs["sdf"]
    zero = sdf * 0.0
    return stack(
        (
            sdf - _sample(sdf, 1, 0, zero=True),
            sdf - _sample(sdf, 0, 1, zero=True),
            zero,
            zero + 1.0,
        ),
        axis=-1,
    )


def _positive_power(base, exponent):
    """GLSL nonnegative pow with finite endpoint subgradients, also for gamma."""
    positive = base.data > 0.0
    return where(positive, where(positive, base, 1.0) ** exponent, 0.0)


def _height_curve(value, params):
    t = _positive_power(_clip(value, 0.0, 1.0), maximum(params["heightGamma"], 0.0001))
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


def _light(context, inputs, params):
    p = {name: _tensor(value) for name, value in params.items()}
    distance = inputs["sdf"] - p["inflate"]
    coverage = 1.0 - _smoothstep(p["inner"], p["outer"] + 0.0001, distance)
    if p["shadeBackground"].item() < 0.5:
        height_map = coverage
    else:
        bevel = maximum(p["bevel"], 0.0001)
        bg_bevel = maximum(
            where(p["backgroundBevel"].data > 0.0, p["backgroundBevel"], bevel),
            0.0001,
        )
        t = p["shadeBackground"] * (
            1.0 - _smoothstep(0.0, bg_bevel, maximum(distance, 0.0))
        )
        height_map = _height_curve(t, p)

    h00 = _sample(height_map, -1, -1)
    h10 = _sample(height_map, 0, -1)
    h20 = _sample(height_map, 1, -1)
    h01 = _sample(height_map, -1, 0)
    h21 = _sample(height_map, 1, 0)
    h02 = _sample(height_map, -1, 1)
    h12 = _sample(height_map, 0, 1)
    h22 = _sample(height_map, 1, 1)
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
    highlight = _positive_power(reflected_z, maximum(p["shininess"], 0.0001))
    highlight = where((lambert.data > 0.0) & (p["specular"].data > 0.0), highlight, 0.0)
    shade = maximum(p["ambient"], 0.0) + maximum(p["diffuse"], 0.0) * lambert
    base = _mix(p["bgColor"], p["fgColor"], coverage[..., None])
    rgb = _clip(
        base[..., :3] * shade[..., None]
        + (maximum(p["specular"], 0.0) * highlight)[..., None],
        0.0,
        1.0,
    )
    lit = _rgba(rgb, base[..., 3])
    slope = _length(gx, gy) * p["normalStrength"]
    height_band = _smoothstep(0.0, 0.08, height_map) * _smoothstep(
        0.0, 0.08, 1.0 - height_map
    )
    influence = height_band * _smoothstep(0.0, 0.08, slope)
    return _mix(base, lit, influence[..., None])


def _noise_rgba(value):
    return Tensor(np.stack((value, value, value, np.ones_like(value)), axis=-1))


def _film_grain(context, inputs, params):
    x, y = context.x.data.astype(np.float32), context.y.data.astype(np.float32)
    angle = x * np.float32(23.14069263277926) + y * np.float32(2.665144142690225)
    value = np.cos(angle) * np.float32(12345.6789)
    return _noise_rgba(value - np.floor(value))


def _perlin_noise(context, inputs, params):
    # Fixed-coordinate noise is deliberately constant in the learning graph.
    # Float32 arithmetic mirrors the shader's hash (float64 changes its cells).
    x = context.x.data.astype(np.float32) / np.float32(256.0) * np.float32(1.5)
    y = context.y.data.astype(np.float32) / np.float32(256.0) * np.float32(1.5)
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy

    def mod289(v):
        return v - np.floor(v * np.float32(1.0 / 289.0)) * np.float32(289.0)

    def permute(v):
        return mod289((v * np.float32(34.0) + np.float32(10.0)) * v)

    def corner(dx, dy):
        index = permute(permute(mod289(ix + dx)) + mod289(iy + dy))
        gx = index * np.float32(1.0 / 41.0)
        gx = (gx - np.floor(gx)) * np.float32(2.0) - np.float32(1.0)
        gy = np.abs(gx) - np.float32(0.5)
        gx = gx - np.floor(gx + np.float32(0.5))
        norm = np.float32(1.79284291400159) - np.float32(0.85373472095314) * (
            gx * gx + gy * gy
        )
        return gx * norm * (fx - dx) + gy * norm * (fy - dy)

    def fade(t):
        return (
            t
            * t
            * t
            * (t * (t * np.float32(6.0) - np.float32(15.0)) + np.float32(10.0))
        )

    u, v = fade(fx), fade(fy)
    lower = corner(0, 0) * (1 - u) + corner(1, 0) * u
    upper = corner(0, 1) * (1 - u) + corner(1, 1) * u
    return _noise_rgba(np.float32(2.3) * (lower * (1 - v) + upper * v))


RULES = {
    "clear_color": _clear_color,
    "fill": _fill,
    "outline": _outline,
    "fill_from_texture": _fill_from_texture,
    "film_grain": _film_grain,
    "isolines": _isolines,
    "light": _light,
    "partial_derivative": _partial_derivative,
    "perlin_noise": _perlin_noise,
}
