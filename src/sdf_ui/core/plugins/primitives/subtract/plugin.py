__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(Plugin("subtract", PluginFamily.PRIMITIVE, TextureKind.SDF, (TextureKind.SDF, TextureKind.SDF), shader=shader("subtract", "plugins/primitives/subtract/shader.glsl"), input_uniforms=("sdf0", "sdf1"), method_of=(TextureKind.SDF,)))
