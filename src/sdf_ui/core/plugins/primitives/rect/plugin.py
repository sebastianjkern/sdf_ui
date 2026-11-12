__docformat__ = "google"

from numbers import Real

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def _uniforms(params):
    radius = params["corner_radius"]
    if isinstance(radius, Real):
        radius = (radius,) * 4
    return {
        "offset": params["center"],
        "size": params["size"],
        "corner_radius": radius,
        "angle": params["angle"],
    }


def register_plugins(registry):
    descriptor = shader("rect", "plugins/primitives/rect/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE)
    registry.register(
        Plugin(
            "rect",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", "size", "corner_radius", "angle"),
            defaults={"corner_radius": 0.0, "angle": 0.0},
            shader=descriptor,
            make_uniforms=_uniforms,
            public=True,
        )
    )
    registry.register(
        Plugin(
            "rounded_rect",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", "size", "corner_radius", "angle"),
            defaults={"angle": 0.0},
            shader=descriptor,
            make_uniforms=_uniforms,
            public=True,
        )
    )
