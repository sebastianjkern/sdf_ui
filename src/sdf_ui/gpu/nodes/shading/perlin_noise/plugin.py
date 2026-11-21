__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import shader

from .implementation import CONTRACT, INTERFACE


def register_plugins(registry):
    descriptor = shader("perlin_noise", "shading/perlin_noise/perlin_noise.wgsl", derivative_contract=CONTRACT, interface=INTERFACE)
    registry.register(
        Plugin(
            "noise",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            shader=descriptor,
            mode="RGB",
            public=True,
        )
    )
    registry.register(
        Plugin(
            "perlin_noise",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            shader=descriptor,
            mode="RGB",
            public=True,
        )
    )
