__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import params, shader


def register_plugins(registry):
    registry.register(Plugin("bezier", PluginFamily.PRIMITIVE, TextureKind.SDF, params=("a", "b", "c"), shader=shader("bezier", "plugins/primitives/bezier/shader.glsl"), make_uniforms=params("a", "b", "c"), public=True))
