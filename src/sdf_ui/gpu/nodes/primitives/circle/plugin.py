

from sdf_ui.gpu.nodes.base import (
    ImageBinding,
    Plugin,
    PluginFamily,
    ShaderInterface,
    TextureKind,
    Uniform,
)
from sdf_ui.gpu.nodes.common import shader

from .implementation import CONTRACT, PROGRAM

_RADIUS_TEXT = 'radius'

__docformat__ = "google"

def _uniforms(params):
    return {"offset": params["center"], _RADIUS_TEXT: params[_RADIUS_TEXT]}


def register_plugins(registry):
    descriptor = shader(
        "circle", "primitives/circle/shader.glsl",
        derivative_contract=CONTRACT,
        program=PROGRAM,
        interface=ShaderInterface(
            output=ImageBinding("destTex", 0, write=True),
            uniforms=(Uniform("offset", "vec2"), Uniform(_RADIUS_TEXT, "float")),
        ),
    )
    registry.register(
        Plugin(
            "circle",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", _RADIUS_TEXT),
            shader=descriptor,
            make_uniforms=_uniforms,
            public=True,
        )
    )
    registry.register(
        Plugin(
            "disc",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", _RADIUS_TEXT),
            shader=descriptor,
            make_uniforms=_uniforms,
            public=True,
        )
    )
