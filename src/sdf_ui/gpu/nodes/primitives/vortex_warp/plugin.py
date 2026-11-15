"""Displace an SDF by a smooth sum of vortex vector fields."""

import numpy as np

from sdf_ui.core.sdf import SDFTexture
from sdf_ui.gpu.nodes._vortex import normalize_vortices, validate_radius, vortex_velocity
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind


def render_vortex_warp(renderer, inputs, params):
    vortices = normalize_vortices(params["vortices"])
    radius = validate_radius(params["radius"])
    strength = float(params["strength"])
    if not np.isfinite(strength):
        raise ValueError("warp strength must be finite")

    source = np.asarray(inputs[0].tex.read(), dtype=np.float32).reshape(
        renderer.ctx.size[1], renderer.ctx.size[0]
    )
    height, width = source.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    vx, vy = vortex_velocity(xx, yy, vortices, radius)
    sample_x = np.clip(xx - vx * strength * radius, 0.0, width - 1.0)
    sample_y = np.clip(yy - vy * strength * radius, 0.0, height - 1.0)

    x0, y0 = np.floor(sample_x).astype(np.int32), np.floor(sample_y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = sample_x - x0, sample_y - y0
    top = source[y0, x0] * (1.0 - fx) + source[y0, x1] * fx
    bottom = source[y1, x0] * (1.0 - fx) + source[y1, x1] * fx
    warped = top * (1.0 - fy) + bottom * fy

    texture = renderer.ctx.r32f()
    texture.write(warped.astype(np.float32).tobytes())
    return SDFTexture(tex=texture, context=renderer.ctx)


def register_plugins(registry):
    registry.register(
        Plugin(
            "vortex_warp",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            params=("vortices", "radius", "strength"),
            defaults={"strength": 0.8},
            render_func=render_vortex_warp,
            method_of=(TextureKind.SDF,),
        )
    )
