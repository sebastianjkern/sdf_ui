__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.common import params, shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "half_plane",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("point", "normal"),
            shader=shader("half_plane", "primitives/half_plane/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            make_uniforms=params("point", "normal"),
            method_of=(TextureKind.SDF,),
            public=True,
        )
    )
