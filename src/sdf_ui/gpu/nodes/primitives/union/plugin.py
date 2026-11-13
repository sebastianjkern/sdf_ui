__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import shader

from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "union",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF, TextureKind.SDF),
            shader=shader("union", "primitives/union/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            input_uniforms=("sdf0", "sdf1"),
            method_of=(TextureKind.SDF,),
        )
    )
