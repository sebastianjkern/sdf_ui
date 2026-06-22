__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import params, shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "ellipse",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", "radii"),
            shader=shader("ellipse", "plugins/primitives/ellipse/shader.glsl"),
            make_uniforms=params("center", "radii"),
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )

