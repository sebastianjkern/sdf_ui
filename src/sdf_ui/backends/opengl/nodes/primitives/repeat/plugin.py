__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.common import shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "repeat",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            params=("s",),
            defaults={"s": 15.0},
            shader=shader("repeat", "primitives/repeat/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            input_uniforms=("sdf0",),
            make_uniforms=lambda p: {"repeat": p["s"]},
            method_of=(TextureKind.SDF,),
        )
    )
