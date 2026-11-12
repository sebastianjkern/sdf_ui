__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.common import params, shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    descriptor = shader(
        "ring", "primitives/ring/shader.glsl",
        derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE,
    )
    for name in ("ring", "annulus"):
        registry.register(
            Plugin(
                name,
                PluginFamily.PRIMITIVE,
                TextureKind.SDF,
                params=("center", "radius", "thickness"),
                defaults={"thickness": 8.0},
                shader=descriptor,
                make_uniforms=params("center", "radius", "thickness"),
                public=True,
                method_of=(TextureKind.SDF,),
            )
        )
