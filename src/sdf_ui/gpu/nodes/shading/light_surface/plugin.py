from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import color, shader

from .implementation import CONTRACT, INTERFACE


def _uniforms(params):
    return {
        "fgColor": color(params["fg_color"]),
        "lightDir": params["light_dir"],
        "ambient": params["ambient"],
        "diffuse": params["diffuse"],
        "specular": params["specular"],
        "shininess": params["shininess"],
    }


def register_plugins(registry):
    registry.register(
        Plugin(
            "light_surface",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            (TextureKind.SDF, TextureKind.HEIGHT),
            params=(
                "fg_color",
                "light_dir",
                "ambient",
                "diffuse",
                "specular",
                "shininess",
            ),
            defaults={
                "fg_color": (1.0, 1.0, 1.0, 1.0),
                "light_dir": (-9 / 20, -13 / 20, 0.7),
                "ambient": 0.35,
                "diffuse": 0.75,
                "specular": 0.2,
                "shininess": 24.0,
            },
            shader=shader(
                "light_surface",
                "shading/light_surface/light_surface.wgsl",
                derivative_contract=CONTRACT,
                interface=INTERFACE,
            ),
            input_uniforms=("sdf", "heightField"),
            make_uniforms=_uniforms,
            mode="RGB",
            method_of=(TextureKind.SDF,),
        )
    )
