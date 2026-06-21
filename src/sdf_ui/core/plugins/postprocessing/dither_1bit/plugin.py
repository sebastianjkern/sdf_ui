__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(Plugin("dither_1bit", PluginFamily.POSTPROCESSING, TextureKind.COLOR, (TextureKind.COLOR,), shader=shader("dither_1bit", "plugins/postprocessing/dither_1bit/shader.glsl"), input_uniforms=("origTex",), method_of=(TextureKind.COLOR,)))
