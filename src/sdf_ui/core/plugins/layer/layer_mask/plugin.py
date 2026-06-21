__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import (
    ensure_rgb,
    rgb_texture_from_array,
    rgba_array,
    shader,
)


def render_layer_mask(renderer, inputs, params):
    import numpy as np

    first = rgba_array(ensure_rgb(renderer, inputs[0])).astype(np.float32)
    second = rgba_array(ensure_rgb(renderer, inputs[1])).astype(np.float32)
    mask = rgba_array(ensure_rgb(renderer, inputs[2])).astype(np.float32)
    alpha = mask[..., 0:1] / 255.0
    return rgb_texture_from_array(renderer.ctx, second * (1.0 - alpha) + first * alpha)


def register_plugins(registry):
    descriptor = shader("layer_mask", "plugins/layer/layer_mask/shader.glsl")
    registry.register(
        Plugin(
            "layer_mask",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.COLOR, TextureKind.COLOR),
            shader=descriptor,
            input_uniforms=("tex0", "tex1", "mask"),
            mode="RGB",
            render_func=render_layer_mask,
            method_of=(TextureKind.COLOR,),
        )
    )
    registry.register(
        Plugin(
            "mask",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.COLOR, TextureKind.COLOR),
            shader=descriptor,
            input_uniforms=("tex0", "tex1", "mask"),
            mode="RGB",
            render_func=render_layer_mask,
            method_of=(TextureKind.COLOR,),
        )
    )
