__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import color, shader


def _uniforms(params):
    return {
        "fgColor": color(params["fg_color"]),
        "bgColor": color(params["bg_color"]),
        "lightDir": params["light_dir"],
        "ambient": params["ambient"],
        "diffuse": params["diffuse"],
        "specular": params["specular"],
        "shininess": params["shininess"],
        "normalStrength": params["normal_strength"],
        "bevel": params["bevel"],
        "height": params["height"],
        "inflate": params["inflate"],
        "inner": params["inner"],
        "outer": params["outer"],
    }


def register_plugins(registry):
    registry.register(
        Plugin(
            "light",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            (TextureKind.SDF,),
            params=(
                "fg_color",
                "bg_color",
                "light_dir",
                "ambient",
                "diffuse",
                "specular",
                "shininess",
                "normal_strength",
                "bevel",
                "height",
                "inflate",
                "inner",
                "outer",
            ),
            defaults={
                "bg_color": (0.0, 0.0, 0.0, 0.0),
                "light_dir": (-0.45, -0.65, 0.7),
                "ambient": 0.35,
                "diffuse": 0.75,
                "specular": 0.2,
                "shininess": 24.0,
                "normal_strength": 4.0,
                "bevel": 8.0,
                "height": 1.0,
                "inflate": 0.0,
                "inner": -1.5,
                "outer": 1.5,
            },
            shader=shader("light", "plugins/shading/light/shader.glsl"),
            input_uniforms=("sdf",),
            make_uniforms=_uniforms,
            mode="RGB",
            method_of=(TextureKind.SDF,),
        )
    )
