__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


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
            method_of=(TextureKind.COLOR,),
        )
    )
