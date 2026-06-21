__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "invert",
            PluginFamily.POSTPROCESSING,
            TextureKind.COLOR,
            (TextureKind.COLOR,),
            shader=shader("invert", "plugins/postprocessing/invert/shader.glsl"),
            input_uniforms=("origTex",),
            method_of=(TextureKind.COLOR,),
        )
    )
