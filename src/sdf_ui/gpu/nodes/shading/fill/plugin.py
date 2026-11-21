

from sdf_ui.gpu.nodes.base import (
    ImageBinding,
    Plugin,
    PluginFamily,
    ShaderInterface,
    TextureKind,
    Uniform,
)
from sdf_ui.gpu.nodes.common import color, shader

from .implementation import CONTRACT

_INFLATE_TEXT = 'inflate'

__docformat__ = "google"

def _uniforms(params):
    return {
        _INFLATE_TEXT: params[_INFLATE_TEXT],
        "background": color(params["bg_color"]),
        "color": color(params["fg_color"]),
        "first": params["inner"],
        "second": params["outer"],
    }


def register_plugins(registry):
    registry.register(
        Plugin(
            "fill",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            (TextureKind.SDF,),
            params=("fg_color", "bg_color", _INFLATE_TEXT, "inner", "outer"),
            defaults={
                "bg_color": (0.0, 0.0, 0.0, 1.0),
                _INFLATE_TEXT: 0.0,
                "inner": -3 / 2,
                "outer": 0.0,
            },
            shader=shader(
                "fill", "shading/fill/fill.wgsl",
                derivative_contract=CONTRACT,
                interface=ShaderInterface(
                    output=ImageBinding("destTex", 0, write=True),
                    inputs=(ImageBinding("sdf", 1, read=True),),
                    uniforms=(
                        Uniform(_INFLATE_TEXT, "float"), Uniform("color", "vec4"),
                        Uniform("background", "vec4"), Uniform("first", "float"),
                        Uniform("second", "float"),
                    ),
                ),
            ),
            input_uniforms=("sdf",),
            make_uniforms=_uniforms,
            method_of=(TextureKind.SDF,),
        )
    )
