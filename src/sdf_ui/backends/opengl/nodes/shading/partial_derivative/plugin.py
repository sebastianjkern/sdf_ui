__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.common import shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "partial_derivative",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            (TextureKind.SDF,),
            shader=shader(
                "partial_derivative", "shading/partial_derivative/shader.glsl",
                derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE,
            ),
            input_uniforms=("sdf",),
            mode="RGB",
            method_of=(TextureKind.SDF,),
        )
    )
