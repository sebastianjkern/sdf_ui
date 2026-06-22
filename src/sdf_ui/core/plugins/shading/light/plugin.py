__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import color, shader

_HEIGHT_PROFILES = {
    "smooth": 0.0,
    "smoothstep": 0.0,
    "cosine": 1.0,
    "soft": 1.0,
    "quad_in": 2.0,
    "sharp": 2.0,
    "quad_out": 3.0,
    "soft_cutout": 3.0,
    "circle_out": 4.0,
    "quarter": 4.0,
    "quarter_circle": 4.0,
    "circle_in": 5.0,
}


def _height_profile(value):
    if isinstance(value, str):
        try:
            return _HEIGHT_PROFILES[value]
        except KeyError as exc:
            known = ", ".join(sorted(_HEIGHT_PROFILES))
            raise ValueError(
                f"Unknown light height profile '{value}'. Known: {known}"
            ) from exc
    return value


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
        "shadeBackground": float(params["shade_background"]),
        "backgroundBevel": params["background_bevel"],
        "heightProfile": _height_profile(params["height_profile"]),
        "heightGamma": params["height_gamma"],
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
                "shade_background",
                "background_bevel",
                "height_profile",
                "height_gamma",
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
                "shade_background": False,
                "background_bevel": 0.0,
                "height_profile": 0.0,
                "height_gamma": 1.0,
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
