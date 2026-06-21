__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import params, shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "line",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("a", "b"),
            shader=shader("line", "plugins/primitives/line/shader.glsl"),
            make_uniforms=params("a", "b"),
            public=True,
        )
    )
