__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import (
    ImageBinding, Plugin, PluginFamily, ShaderInterface, TextureKind, Uniform,
)
from sdf_ui.backends.opengl.nodes.common import color, shader
from .implementation import CONTRACT, PROGRAM


def _uniforms(params):
    return {
        "inflate": params["inflate"],
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
            params=("fg_color", "bg_color", "inflate", "inner", "outer"),
            defaults={
                "bg_color": (0.0, 0.0, 0.0, 1.0),
                "inflate": 0.0,
                "inner": -1.5,
                "outer": 0.0,
            },
            shader=shader(
                "fill", "shading/fill/shader.glsl",
                derivative_contract=CONTRACT,
                program=PROGRAM,
                interface=ShaderInterface(
                    output=ImageBinding("destTex", 0, write=True),
                    inputs=(ImageBinding("sdf", 1, read=True),),
                    uniforms=(
                        Uniform("inflate", "float"), Uniform("color", "vec4"),
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
