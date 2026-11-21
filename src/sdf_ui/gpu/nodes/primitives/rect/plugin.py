

from numbers import Real

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import shader

from .implementation import CONTRACT, INTERFACE

_ANGLE_TEXT = 'angle'

_CORNER_RADIUS_TEXT = 'corner_radius'

__docformat__ = "google"

def _uniforms(params):
    radius = params[_CORNER_RADIUS_TEXT]
    if isinstance(radius, Real):
        radius = (radius,) * 4
    return {
        "offset": params["center"],
        "size": params["size"],
        _CORNER_RADIUS_TEXT: radius,
        _ANGLE_TEXT: params[_ANGLE_TEXT],
    }


def register_plugins(registry):
    descriptor = shader("rect", "primitives/rect/rect.wgsl", derivative_contract=CONTRACT, interface=INTERFACE)
    registry.register(
        Plugin(
            "rect",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", "size", _CORNER_RADIUS_TEXT, _ANGLE_TEXT),
            defaults={_CORNER_RADIUS_TEXT: 0.0, _ANGLE_TEXT: 0.0},
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
            params=("center", "size", _CORNER_RADIUS_TEXT, _ANGLE_TEXT),
            defaults={_ANGLE_TEXT: 0.0},
            shader=descriptor,
            make_uniforms=_uniforms,
            public=True,
        )
    )
