__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import params, shader

from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "arc",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", "radius", "start_angle", "end_angle"),
            shader=shader("arc", "primitives/arc/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            make_uniforms=params("center", "radius", "start_angle", "end_angle"),
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )
