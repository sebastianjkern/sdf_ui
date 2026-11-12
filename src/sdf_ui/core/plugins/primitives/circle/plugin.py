__docformat__ = "google"

from sdf_ui.core.plugins.base import (
    ImageBinding, Plugin, PluginFamily, ShaderInterface, TextureKind, Uniform,
)
from sdf_ui.core.plugins.common import shader
from .implementation import CONTRACT, PROGRAM


def _uniforms(params):
    return {"offset": params["center"], "radius": params["radius"]}


def register_plugins(registry):
    descriptor = shader(
        "circle", "plugins/primitives/circle/shader.glsl",
        derivative_contract=CONTRACT,
        program=PROGRAM,
        interface=ShaderInterface(
            output=ImageBinding("destTex", 0, write=True),
            uniforms=(Uniform("offset", "vec2"), Uniform("radius", "float")),
        ),
    )
    registry.register(
        Plugin(
            "circle",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", "radius"),
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
            params=("center", "radius"),
            shader=descriptor,
            make_uniforms=_uniforms,
            public=True,
        )
    )
