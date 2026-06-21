__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import params, shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "transparency",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR,),
            params=("alpha",),
            shader=shader("transparency", "plugins/layer/transparency/shader.glsl"),
            input_uniforms=("tex0",),
            make_uniforms=params("alpha"),
            method_of=(TextureKind.COLOR,),
        )
    )
