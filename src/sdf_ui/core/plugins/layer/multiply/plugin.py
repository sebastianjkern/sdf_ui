__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "multiply",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.COLOR),
            shader=shader("multiply", "plugins/layer/multiply/shader.glsl"),
            input_uniforms=("mask1", "mask2"),
            method_of=(TextureKind.COLOR,),
        )
    )
