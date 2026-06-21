__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    registry.register(Plugin("partial_derivative", PluginFamily.SHADING, TextureKind.COLOR, (TextureKind.SDF,), shader=shader("partial_derivative", "plugins/shading/partial_derivative/shader.glsl"), input_uniforms=("sdf",), mode="RGB", method_of=(TextureKind.SDF,)))
