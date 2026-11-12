__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "dither_1bit",
            PluginFamily.POSTPROCESSING,
            TextureKind.COLOR,
            (TextureKind.COLOR,),
            shader=shader(
                "dither_1bit", "plugins/postprocessing/dither_1bit/shader.glsl",
                derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE,
            ),
            input_uniforms=("origTex",),
            method_of=(TextureKind.COLOR,),
        )
    )
