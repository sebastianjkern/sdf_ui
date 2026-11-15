"""Diagnostic color mapping for scalar distance fields."""

import numpy as np

from sdf_ui.core.color import ColorSpaceMode, ColorTexture
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind


def render_field_debug(renderer, inputs, params):
    field = np.asarray(inputs[0].tex.read(), dtype=np.float32).reshape(
        renderer.ctx.size[1], renderer.ctx.size[0]
    )
    span = float(params["range"])
    if not np.isfinite(span) or span <= 0:
        raise ValueError("range must be a positive finite number")
    field = np.nan_to_num(field, nan=span, posinf=span, neginf=-span)
    # Negative (inside) values move from blue through a pale zero contour;
    # positive (outside) values move toward orange.
    stops = np.asarray(
        (
            (0.12, 0.25, 0.68),
            (0.34, 0.72, 0.83),
            (0.94, 0.94, 0.82),
            (0.95, 0.57, 0.28),
            (0.65, 0.18, 0.12),
        ),
        dtype=np.float32,
    )
    normalized = np.clip(field / span, -1.0, 1.0)
    position = (normalized + 1.0) * 2.0
    lower = np.minimum(position.astype(np.int32), 3)
    fraction = (position - lower)[..., None]
    rgb = stops[lower] * (1.0 - fraction) + stops[lower + 1] * fraction
    alpha = np.ones((*field.shape, 1), dtype=np.float32)
    texture = renderer.ctx.rgba8()
    pixels = np.rint(np.concatenate((rgb, alpha), axis=-1) * 255).astype(np.uint8)
    texture.write(pixels.tobytes())
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
