__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "grid",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("offset", "size", "angle"),
            defaults={"angle": 0.0},
            shader=shader("grid", "plugins/primitives/grid/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            make_uniforms=lambda p: {
                "offset": p["offset"],
                "grid_size": p["size"],
                "angle": p["angle"],
            },
            public=True,
        )
    )
