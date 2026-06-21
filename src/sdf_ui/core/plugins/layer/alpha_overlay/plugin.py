__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "alpha_overlay",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.COLOR),
            shader=shader("overlay", "plugins/layer/alpha_overlay/shader.glsl"),
            input_uniforms=("tex1", "tex0"),
            method_of=(TextureKind.COLOR,),
        )
    )
