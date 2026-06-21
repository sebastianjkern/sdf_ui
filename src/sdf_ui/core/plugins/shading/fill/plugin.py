__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def _uniforms(params):
    return {
        "inflate": params["inflate"],
        "background": params["bg_color"],
        "color": params["fg_color"],
        "first": params["inner"],
        "second": params["outer"],
    }


def register_plugins(registry):
    registry.register(Plugin("fill", PluginFamily.SHADING, TextureKind.COLOR, (TextureKind.SDF,), params=("fg_color", "bg_color", "inflate", "inner", "outer"), defaults={"bg_color": (0.0, 0.0, 0.0, 1.0), "inflate": 0.0, "inner": -1.5, "outer": 0.0}, shader=shader("fill", "plugins/shading/fill/shader.glsl"), input_uniforms=("sdf",), make_uniforms=_uniforms, method_of=(TextureKind.SDF,)))
