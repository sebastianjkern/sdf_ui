__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "abs",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            shader=shader("abs", "plugins/primitives/abs/shader.glsl"),
            input_uniforms=("sdf0",),
            method_of=(TextureKind.SDF,),
        )
    )
