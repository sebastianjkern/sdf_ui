__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.common import colors, shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    descriptor = shader(
        "clear_color", "shading/clear_color/shader.glsl",
        derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE,
    )
    registry.register(
        Plugin(
            "clear",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            params=("color",),
            shader=descriptor,
            make_uniforms=colors("color"),
            public=True,
        )
    )
    registry.register(
        Plugin(
            "clear_color",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            params=("color",),
            shader=descriptor,
            make_uniforms=colors("color"),
            public=True,
        )
    )
