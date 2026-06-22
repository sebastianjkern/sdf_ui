__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import params, shader


def register_plugins(registry):
    descriptor = shader("capsule", "plugins/primitives/capsule/shader.glsl")
    for name in ("capsule", "segment"):
        registry.register(
            Plugin(
                name,
                PluginFamily.PRIMITIVE,
                TextureKind.SDF,
                params=("a", "b", "radius"),
                shader=descriptor,
                make_uniforms=params("a", "b", "radius"),
                public=True,
            )
        )

