__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import params, shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "half_plane",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("point", "normal"),
            shader=shader("half_plane", "plugins/primitives/half_plane/shader.glsl"),
            make_uniforms=params("point", "normal"),
            method_of=(TextureKind.SDF,),
            public=True,
        )
    )
