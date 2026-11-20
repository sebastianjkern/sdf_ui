"""GPU visualizations of vector direction and magnitude."""

__docformat__ = "google"

from sdf_ui.core.color import ColorSpaceMode
from sdf_ui.gpu.nodes.base import (
    ImageBinding,
    Plugin,
    PluginFamily,
    ShaderInterface,
    TextureKind,
    Uniform,
)
from sdf_ui.gpu.nodes.common import shader

from .implementation import CONTOURS, VISUALIZE

_SCALE = "scale"
_MAGNITUDE = "magnitude"
_SPACING = "spacing"
_LINE_WIDTH = "line_width"
_FLOAT = "float"


def _visualize_uniforms(params):
    import math

    scale, magnitude = float(params[_SCALE]), float(params[_MAGNITUDE])
    if not math.isfinite(scale) or not math.isfinite(magnitude) or magnitude < 0:
        raise ValueError("scale must be finite and magnitude must be finite and nonnegative")
    return {_SCALE: scale, _MAGNITUDE: magnitude}


def _contour_uniforms(params):
    import math

    values = {
        _SPACING: float(params[_SPACING]),
        _LINE_WIDTH: float(params[_LINE_WIDTH]),
        _SCALE: float(params[_SCALE]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("contour parameters must be finite")
    if values[_SPACING] <= 0 or values[_LINE_WIDTH] < 0:
        raise ValueError("spacing must be positive and line_width must be nonnegative")
    return values


def register_plugins(registry):
    registry.register(
        Plugin(
            "field_visualize", PluginFamily.SHADING, TextureKind.COLOR,
            (TextureKind.FIELD,), params=(_SCALE, _MAGNITUDE),
            defaults={_SCALE: 1.0, _MAGNITUDE: 0.04},
            shader=shader(
                "field_visualize", "shading/field_visualize/visualize.glsl",
                derivative_contract=VISUALIZE,
                interface=ShaderInterface(
                    ImageBinding("destTex", 0, write=True),
                    (ImageBinding("fieldTex", 1, read=True),),
                    (Uniform(_SCALE, _FLOAT), Uniform(_MAGNITUDE, _FLOAT)),
                ),
            ),
            make_uniforms=_visualize_uniforms,
            input_uniforms=("fieldTex",), mode=ColorSpaceMode.RGB,
            method_of=(TextureKind.FIELD,),
        )
    )
    registry.register(
        Plugin(
            "field_contours", PluginFamily.SHADING, TextureKind.COLOR,
            (TextureKind.FIELD,), params=(_SPACING, _LINE_WIDTH, _SCALE),
            defaults={_SPACING: 8.0, _LINE_WIDTH: 1.0, _SCALE: 1.0},
            shader=shader(
                "field_contours", "shading/field_visualize/contours.glsl",
                derivative_contract=CONTOURS,
                interface=ShaderInterface(
                    ImageBinding("destTex", 0, write=True),
                    (ImageBinding("fieldTex", 1, read=True),),
                    (Uniform(_SPACING, _FLOAT), Uniform(_LINE_WIDTH, _FLOAT), Uniform(_SCALE, _FLOAT)),
                ),
            ),
            make_uniforms=_contour_uniforms,
            input_uniforms=("fieldTex",), mode=ColorSpaceMode.RGB,
            method_of=(TextureKind.FIELD,),
        )
    )
