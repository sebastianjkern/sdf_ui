__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import params, shader

from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "bezier",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("a", "b", "c"),
            shader=shader("bezier", "primitives/bezier/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            make_uniforms=params("a", "b", "c"),
            public=True,
        )
    )
