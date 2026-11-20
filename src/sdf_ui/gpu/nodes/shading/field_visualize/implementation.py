"""Field visualization output contracts and CPU reference mappings."""

import numpy as np

from sdf_ui.gpu.nodes.differentiation import ShaderArgument, ShaderDifferentiation

_SCALE = "scale"
_FLOAT_TYPE = "float"
_CONTOUR_HALF_WIDTH = 0.75
_CONTOUR_RED_BASE = 0.035
_CONTOUR_RED_RANGE = 0.235
_CONTOUR_GREEN_BASE = 0.055
_CONTOUR_GREEN_RANGE = 0.825
_CONTOUR_BLUE_BASE = 0.095
_CONTOUR_BLUE_RANGE = 0.825


def _hsv_to_rgb(hue, value):
    h = (hue % 1.0) * 6.0
    sector = np.floor(h).astype(np.int32)
    f = h - sector
    saturation = 0.78
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * f)
    t = value * (1.0 - saturation * (1.0 - f))
    sector = sector % 6
    r = np.choose(sector, (value, q, p, p, t, value))
    g = np.choose(sector, (t, value, value, q, p, p))
    b = np.choose(sector, (p, p, t, value, value, q))
    return np.stack((r, g, b, np.ones_like(r)), axis=-1)


def visualize_reference(context, inputs, params):
    from sdf_ui.learning.autodiff import Tensor

    field = inputs["fieldTex"].data
    scale = float(params[_SCALE].item())
    magnitude_scale = float(params["magnitude"].item())
    vx = field[..., 0] * scale
    vy = field[..., 1] * scale
    magnitude = np.sqrt(vx * vx + vy * vy)
    hue = np.arctan2(vy, vx) / (2.0 * np.pi) + 0.5
    value = 1.0 - np.exp(-magnitude * magnitude_scale)
    rgba = _hsv_to_rgb(hue, value)
    return Tensor(rgba)


def contours_reference(context, inputs, params):
    from sdf_ui.learning.autodiff import Tensor

    vectors = inputs["fieldTex"].data
    magnitude = (
        np.sqrt(vectors[..., 0] ** 2 + vectors[..., 1] ** 2)
        * float(params[_SCALE].item())
    )
    spacing = float(params["spacing"].item())
    distance = np.abs(np.mod(magnitude + spacing * 0.5, spacing) - spacing * 0.5)
    width = float(params["line_width"].item())
    edge0, edge1 = max(width - _CONTOUR_HALF_WIDTH, 0.0), width + _CONTOUR_HALF_WIDTH
    amount = np.clip((distance - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    smooth = amount * amount * (3.0 - 2.0 * amount)
    line = 1.0 - smooth
    rgba = np.stack(
        (_CONTOUR_RED_BASE + _CONTOUR_RED_RANGE * line,
         _CONTOUR_GREEN_BASE + _CONTOUR_GREEN_RANGE * line,
         _CONTOUR_BLUE_BASE + _CONTOUR_BLUE_RANGE * line, np.ones_like(line)),
        axis=-1,
    )
    return Tensor(rgba)


VISUALIZE = ShaderDifferentiation(
    "field_visualize", "gpu/nodes/shading/field_visualize/visualize.glsl",
    (ShaderArgument("fieldTex", (4,), "rgba32float"),),
    (ShaderArgument(_SCALE, (), _FLOAT_TYPE), ShaderArgument("magnitude", (), _FLOAT_TYPE)),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    zero_gradient_parameters=(_SCALE, "magnitude"), behavior="constant",
    notes="Diagnostic visualization with no parameter gradients.",
    reference=visualize_reference,
)
CONTOURS = ShaderDifferentiation(
    "field_contours", "gpu/nodes/shading/field_visualize/contours.glsl",
    (ShaderArgument("fieldTex", (4,), "rgba32float"),),
    (ShaderArgument("spacing", (), _FLOAT_TYPE), ShaderArgument("line_width", (), _FLOAT_TYPE), ShaderArgument(_SCALE, (), _FLOAT_TYPE)),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    zero_gradient_parameters=("spacing", "line_width", _SCALE), behavior="constant",
    notes="Diagnostic contours with no parameter gradients.",
    reference=contours_reference,
)
