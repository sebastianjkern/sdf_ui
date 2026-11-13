__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import params, shader

from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    registry.register(
        Plugin(
            "interpolate",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF, TextureKind.SDF),
            params=("t",),
            defaults={"t": 0.5},
            shader=shader(
                "interpolation", "primitives/interpolate/shader.glsl",
                derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE,
            ),
            input_uniforms=("sdf0", "sdf1"),
            make_uniforms=params("t"),
            method_of=(TextureKind.SDF,),
        )
    )
