__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import params, shader


def register_plugins(registry):
    descriptor = shader("clear_color", "plugins/shading/clear_color/shader.glsl")
    registry.register(Plugin("clear", PluginFamily.SHADING, TextureKind.COLOR, params=("color",), shader=descriptor, make_uniforms=params("color"), public=True))
    registry.register(Plugin("clear_color", PluginFamily.SHADING, TextureKind.COLOR, params=("color",), shader=descriptor, make_uniforms=params("color"), public=True))
