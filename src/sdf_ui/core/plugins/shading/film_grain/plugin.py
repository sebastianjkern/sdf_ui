__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "film_grain",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            shader=shader("film_grain", "plugins/shading/film_grain/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            mode="RGB",
            public=True,
        )
    )
