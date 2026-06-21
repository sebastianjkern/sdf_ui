__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(Plugin("to_lab", PluginFamily.POSTPROCESSING, TextureKind.COLOR, (TextureKind.COLOR,), shader=shader("to_lab", "plugins/postprocessing/to_lab/shader.glsl"), input_uniforms=("origTex",), mode="LAB", method_of=(TextureKind.COLOR,)))
