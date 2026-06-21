__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import (
    ensure_rgb,
    rgb_texture_from_array,
    rgba_array,
    shader,
)


def render_multiply(renderer, inputs, params):
    import numpy as np

    first = rgba_array(ensure_rgb(renderer, inputs[0])).astype(np.float32) / 255.0
    second = rgba_array(ensure_rgb(renderer, inputs[1])).astype(np.float32) / 255.0
    return rgb_texture_from_array(renderer.ctx, first * second * 255.0)


def register_plugins(registry):
    registry.register(
        Plugin(
            "multiply",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.COLOR),
            shader=shader("multiply", "plugins/layer/multiply/shader.glsl"),
            input_uniforms=("mask1", "mask2"),
            mode="RGB",
            render_func=render_multiply,
            method_of=(TextureKind.COLOR,),
        )
    )
