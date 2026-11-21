__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import shader

from .implementation import CONTRACT, INTERFACE


def register_plugins(registry):
    registry.register(
        Plugin(
            "subtract",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF, TextureKind.SDF),
            shader=shader(
                "subtract",
                "primitives/subtract/subtract.wgsl",
                derivative_contract=CONTRACT,
                interface=INTERFACE,
            ),
            input_uniforms=("sdf0", "sdf1"),
            method_of=(TextureKind.SDF,),
        )
    )
