__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "triangle",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("point_1", "point_2", "point_3"),
            shader=shader("triangle", "plugins/primitives/triangle/shader.glsl"),
            make_uniforms=lambda p: {
                "point0": p["point_1"],
                "point1": p["point_2"],
                "point2": p["point_3"],
            },
            public=True,
        )
    )
