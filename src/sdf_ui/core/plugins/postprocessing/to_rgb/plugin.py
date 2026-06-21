__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "to_rgb",
            PluginFamily.POSTPROCESSING,
            TextureKind.COLOR,
            (TextureKind.COLOR,),
            shader=shader("to_rgb", "plugins/postprocessing/to_rgb/shader.glsl"),
            input_uniforms=("origTex",),
            mode="RGB",
            method_of=(TextureKind.COLOR,),
        )
    )
