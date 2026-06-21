__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def _uniforms(params):
    return {"offset": params["center"], "radius": params["radius"]}


def register_plugins(registry):
    descriptor = shader("circle", "plugins/primitives/circle/shader.glsl")
    registry.register(
        Plugin(
            "circle",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", "radius"),
            shader=descriptor,
            make_uniforms=_uniforms,
            public=True,
        )
    )
    registry.register(
        Plugin(
            "disc",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", "radius"),
            shader=descriptor,
            make_uniforms=_uniforms,
            public=True,
        )
    )
