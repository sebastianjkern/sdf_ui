__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import params, shader

from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "ellipse",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", "radii"),
            shader=shader("ellipse", "primitives/ellipse/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            make_uniforms=params("center", "radii"),
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )
