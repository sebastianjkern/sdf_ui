__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def _uniforms(params):
    return {
        "offset": params["center"],
        "size": params["size"],
        "corner_radius": params["corner_radius"],
        "angle": params["angle"],
    }


def register_plugins(registry):
    descriptor = shader("rect", "plugins/primitives/rect/shader.glsl")
    registry.register(Plugin("rect", PluginFamily.PRIMITIVE, TextureKind.SDF, params=("center", "size", "corner_radius", "angle"), defaults={"corner_radius": 0.0, "angle": 0.0}, shader=descriptor, make_uniforms=_uniforms, public=True))
    registry.register(Plugin("rounded_rect", PluginFamily.PRIMITIVE, TextureKind.SDF, params=("center", "size", "corner_radius", "angle"), defaults={"angle": 0.0}, shader=descriptor, make_uniforms=_uniforms, public=True))
