__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import params, shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "arc",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", "radius", "start_angle", "end_angle"),
            shader=shader("arc", "plugins/primitives/arc/shader.glsl"),
            make_uniforms=params("center", "radius", "start_angle", "end_angle"),
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )

