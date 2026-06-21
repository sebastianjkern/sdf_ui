__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "film_grain",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            shader=shader("film_grain", "plugins/shading/film_grain/shader.glsl"),
            mode="RGB",
            public=True,
        )
    )
