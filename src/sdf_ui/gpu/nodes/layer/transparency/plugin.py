__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import params, shader

from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "transparency",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR,),
            params=("alpha",),
            shader=shader("transparency", "layer/transparency/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            input_uniforms=("tex0",),
            make_uniforms=params("alpha"),
            method_of=(TextureKind.COLOR,),
        )
    )
