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


def _visualize_uniforms(params):
    import math

    scale, magnitude = float(params["scale"]), float(params["magnitude"])
    if not math.isfinite(scale) or not math.isfinite(magnitude) or magnitude < 0:
        raise ValueError("scale must be finite and magnitude must be finite and nonnegative")
    return {"scale": scale, "magnitude": magnitude}


def _contour_uniforms(params):
    import math

    values = {
        "spacing": float(params["spacing"]),
        "line_width": float(params["line_width"]),
        "scale": float(params["scale"]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("contour parameters must be finite")
    if values["spacing"] <= 0 or values["line_width"] < 0:
        raise ValueError("spacing must be positive and line_width must be nonnegative")
    return values


def register_plugins(registry):
    registry.register(
        Plugin(
            "field_visualize", PluginFamily.SHADING, TextureKind.COLOR,
            (TextureKind.FIELD,), params=("scale", "magnitude"),
            defaults={"scale": 1.0, "magnitude": 0.04},
            shader=shader(
                "field_visualize", "shading/field_visualize/visualize.glsl",
                derivative_contract=VISUALIZE,
                interface=ShaderInterface(
                    ImageBinding("destTex", 0, write=True),
                    (ImageBinding("fieldTex", 1, read=True),),
                    (Uniform("scale", "float"), Uniform("magnitude", "float")),
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
            (TextureKind.FIELD,), params=("spacing", "line_width", "scale"),
            defaults={"spacing": 8.0, "line_width": 1.0, "scale": 1.0},
            shader=shader(
                "field_contours", "shading/field_visualize/contours.glsl",
                derivative_contract=CONTOURS,
                interface=ShaderInterface(
                    ImageBinding("destTex", 0, write=True),
                    (ImageBinding("fieldTex", 1, read=True),),
                    (Uniform("spacing", "float"), Uniform("line_width", "float"), Uniform("scale", "float")),
                ),
            ),
            make_uniforms=_contour_uniforms,
            input_uniforms=("fieldTex",), mode=ColorSpaceMode.RGB,
            method_of=(TextureKind.FIELD,),
        )
    )
