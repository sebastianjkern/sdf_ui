"""Diagnostic color mapping for scalar distance fields."""

import numpy as np

from sdf_ui.core.color import ColorSpaceMode, ColorTexture
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind


def _distance_field(renderer, input_texture, span):
    if not np.isfinite(span) or span <= 0:
        raise ValueError("range must be a positive finite number")
    values = np.asarray(input_texture.read(), dtype=np.float32).reshape(
        renderer.ctx.size[1], renderer.ctx.size[0]
    )
    return np.nan_to_num(values, nan=span, posinf=span, neginf=-span), span


def _debug_rgba(field, span):
    stops = np.asarray(
        ((0.12, 0.25, 0.68), (0.34, 0.72, 0.83), (0.94, 0.94, 0.82),
         (0.95, 0.57, 0.28), (0.65, 0.18, 0.12)),
        dtype=np.float32,
    )
    position = (np.clip(field / span, -1.0, 1.0) + 1.0) * 2.0
    lower = np.minimum(position.astype(np.int32), 3)
    fraction = (position - lower)[..., None]
    rgb = stops[lower] * (1.0 - fraction) + stops[lower + 1] * fraction
    alpha = np.ones((*field.shape, 1), dtype=np.float32)
    return np.rint(np.concatenate((rgb, alpha), axis=-1) * 255).astype(np.uint8)


def render_field_debug(renderer, inputs, params):
    field, span = _distance_field(renderer, inputs[0].tex, float(params["range"]))
    texture = renderer.ctx.rgba8()
    texture.write(_debug_rgba(field, span).tobytes())
    return ColorTexture(tex=texture, context=renderer.ctx, mode=ColorSpaceMode.RGB)


def register_plugins(registry):
    registry.register(
        Plugin(
            "field_debug",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            (TextureKind.SDF,),
            params=("range",),
            defaults={"range": 32.0},
            render_func=render_field_debug,
            mode=ColorSpaceMode.RGB,
        )
    )
