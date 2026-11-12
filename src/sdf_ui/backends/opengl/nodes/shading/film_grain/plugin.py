__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.common import shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "film_grain",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            shader=shader("film_grain", "shading/film_grain/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            mode="RGB",
            public=True,
        )
    )
