__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import shader

from .implementation import CONTRACT, INTERFACE


def register_plugins(registry):
    registry.register(
        Plugin(
            "partial_derivative",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            (TextureKind.SDF,),
            shader=shader(
                "partial_derivative", "shading/partial_derivative/partial_derivative.wgsl",
                derivative_contract=CONTRACT, interface=INTERFACE,
            ),
            input_uniforms=("sdf",),
            mode="RGB",
            method_of=(TextureKind.SDF,),
        )
    )
