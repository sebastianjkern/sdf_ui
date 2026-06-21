__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(Plugin("outline", PluginFamily.SHADING, TextureKind.COLOR, (TextureKind.SDF,), params=("fg_color", "bg_color", "inflate"), defaults={"bg_color": (0.0, 0.0, 0.0, 0.0), "inflate": 0.0}, shader=shader("outline", "plugins/shading/outline/shader.glsl"), input_uniforms=("sdf",), make_uniforms=lambda p: {"background": p["bg_color"], "outline": p["fg_color"], "inflate": p["inflate"]}, method_of=(TextureKind.SDF,)))
