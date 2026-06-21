__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "repeat",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            params=("s",),
            defaults={"s": 15.0},
            shader=shader("repeat", "plugins/primitives/repeat/shader.glsl"),
            input_uniforms=("sdf0",),
            make_uniforms=lambda p: {"repeat": p["s"]},
            method_of=(TextureKind.SDF,),
        )
    )
