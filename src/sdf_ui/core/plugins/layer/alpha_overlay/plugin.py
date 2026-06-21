__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import (
    ensure_rgb,
    rgb_texture_from_array,
    rgba_array,
    shader,
)


def render_alpha_overlay(renderer, inputs, params):
    import numpy as np

    base = rgba_array(ensure_rgb(renderer, inputs[0])).astype(np.float32) / 255.0
    top = rgba_array(ensure_rgb(renderer, inputs[1])).astype(np.float32) / 255.0

    alpha = top[..., 3:4] + (1.0 - top[..., 3:4]) * base[..., 3:4]
    rgb = np.zeros_like(top[..., :3])
    np.divide(
        top[..., :3] * top[..., 3:4]
        + base[..., :3] * (1.0 - top[..., 3:4]) * base[..., 3:4],
        alpha,
        out=rgb,
        where=alpha > 0.0,
    )
    return rgb_texture_from_array(renderer.ctx, np.dstack((rgb, alpha[..., 0])) * 255.0)


def register_plugins(registry):
    registry.register(
        Plugin(
            "alpha_overlay",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.COLOR),
            shader=shader("overlay", "plugins/layer/alpha_overlay/shader.glsl"),
            input_uniforms=("tex1", "tex0"),
            mode="RGB",
            render_func=render_alpha_overlay,
            method_of=(TextureKind.COLOR,),
        )
    )
