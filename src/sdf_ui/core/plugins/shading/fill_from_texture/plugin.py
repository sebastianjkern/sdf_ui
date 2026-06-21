__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import params, shader


def register_plugins(registry):
    registry.register(
        Plugin(
            "fill_from_texture",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            (TextureKind.SDF, TextureKind.COLOR),
            params=("background", "inflate"),
            defaults={"background": (0.0, 0.0, 0.0, 0.0), "inflate": 0},
            shader=shader(
                "fill_from_texture", "plugins/shading/fill_from_texture/shader.glsl"
            ),
            input_uniforms=("sdf", "origTex"),
            make_uniforms=params("background", "inflate"),
            method_of=(TextureKind.SDF,),
        )
    )
