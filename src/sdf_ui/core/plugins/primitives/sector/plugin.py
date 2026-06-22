__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import params, shader


def register_plugins(registry):
    descriptor = shader("sector", "plugins/primitives/sector/shader.glsl")
    for name in ("sector", "wedge", "pie"):
        registry.register(
            Plugin(
                name,
                PluginFamily.PRIMITIVE,
                TextureKind.SDF,
                params=("center", "radius", "start_angle", "end_angle"),
                shader=descriptor,
                make_uniforms=params("center", "radius", "start_angle", "end_angle"),
                public=True,
                method_of=(TextureKind.SDF,),
            )
        )

