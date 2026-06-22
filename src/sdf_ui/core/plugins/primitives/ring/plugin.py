__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import params, shader


def register_plugins(registry):
    descriptor = shader("ring", "plugins/primitives/ring/shader.glsl")
    for name in ("ring", "annulus"):
        registry.register(
            Plugin(
                name,
                PluginFamily.PRIMITIVE,
                TextureKind.SDF,
                params=("center", "radius", "thickness"),
                defaults={"thickness": 8.0},
                shader=descriptor,
                make_uniforms=params("center", "radius", "thickness"),
                public=True,
                method_of=(TextureKind.SDF,),
            )
        )

