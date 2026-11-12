__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.common import shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    descriptor = shader("perlin_noise", "shading/perlin_noise/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE)
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
