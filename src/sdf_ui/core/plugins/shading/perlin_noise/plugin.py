__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def register_plugins(registry):
    descriptor = shader("perlin_noise", "plugins/shading/perlin_noise/shader.glsl")
    registry.register(Plugin("noise", PluginFamily.SHADING, TextureKind.COLOR, shader=descriptor, mode="RGB", public=True))
    registry.register(Plugin("perlin_noise", PluginFamily.SHADING, TextureKind.COLOR, shader=descriptor, mode="RGB", public=True))
