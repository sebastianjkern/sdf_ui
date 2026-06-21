__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import params, shader


def register_plugins(registry):
    registry.register(Plugin("interpolate", PluginFamily.PRIMITIVE, TextureKind.SDF, (TextureKind.SDF, TextureKind.SDF), params=("t",), defaults={"t": 0.5}, shader=shader("interpolation", "plugins/primitives/interpolate/shader.glsl"), input_uniforms=("sdf0", "sdf1"), make_uniforms=params("t"), method_of=(TextureKind.SDF,)))
