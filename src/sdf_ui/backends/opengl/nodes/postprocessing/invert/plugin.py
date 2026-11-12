__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.common import shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "invert",
            PluginFamily.POSTPROCESSING,
            TextureKind.COLOR,
            (TextureKind.COLOR,),
            shader=shader("invert", "postprocessing/invert/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            input_uniforms=("origTex",),
            method_of=(TextureKind.COLOR,),
        )
    )
