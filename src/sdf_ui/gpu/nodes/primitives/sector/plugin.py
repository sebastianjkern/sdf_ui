__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import params, shader

from .implementation import CONTRACT, INTERFACE, PROGRAM


def register_plugins(registry):
    descriptor = shader("sector", "primitives/sector/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE)
    for name in ("sector", "wedge", "pie"):
        registry.register(
            Plugin(
                name,
                PluginFamily.PRIMITIVE,
                TextureKind.SDF,
                params=("center", "radius", "start_angle", "end_angle"),
                shader=descriptor,
                make_uniforms=params("center", "radius", "start_angle", "end_angle"),
                public=True,
                method_of=(TextureKind.SDF,),
            )
        )
