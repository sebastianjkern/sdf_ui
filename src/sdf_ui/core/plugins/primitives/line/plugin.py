__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import params, shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "line",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("a", "b"),
            shader=shader("line", "plugins/primitives/line/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            make_uniforms=params("a", "b"),
            public=True,
        )
    )
