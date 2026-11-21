__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import shader

from .implementation import CONTRACT, INTERFACE


def register_plugins(registry):
    registry.register(
        Plugin(
            "film_grain",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            shader=shader("film_grain", "shading/film_grain/film_grain.wgsl", derivative_contract=CONTRACT, interface=INTERFACE),
            mode="RGB",
            public=True,
        )
    )
