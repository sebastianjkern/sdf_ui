__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(Plugin("smooth_union", PluginFamily.PRIMITIVE, TextureKind.SDF, (TextureKind.SDF, TextureKind.SDF), params=("k",), defaults={"k": 0.025}, shader=shader("smooth_min", "plugins/primitives/smooth_union/shader.glsl"), input_uniforms=("sdf0", "sdf1"), make_uniforms=lambda p: {"smoothness": p["k"]}, method_of=(TextureKind.SDF,)))
