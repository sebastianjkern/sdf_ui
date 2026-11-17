"""Field visualization output contracts and CPU reference mappings."""

import numpy as np

from sdf_ui.gpu.nodes.differentiation import ShaderArgument, ShaderDifferentiation


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
    scale = float(params["scale"].item())
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
        * float(params["scale"].item())
    )
    spacing = float(params["spacing"].item())
    distance = np.abs(np.mod(magnitude + spacing * 0.5, spacing) - spacing * 0.5)
    width = float(params["line_width"].item())
    edge0, edge1 = max(width - 0.75, 0.0), width + 0.75
    amount = np.clip((distance - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    smooth = amount * amount * (3.0 - 2.0 * amount)
    line = 1.0 - smooth
    rgba = np.stack(
        (0.035 + 0.235 * line, 0.055 + 0.825 * line, 0.095 + 0.825 * line, np.ones_like(line)),
        axis=-1,
    )
    return Tensor(rgba)


VISUALIZE = ShaderDifferentiation(
    "field_visualize", "gpu/nodes/shading/field_visualize/visualize.glsl",
    (ShaderArgument("fieldTex", (4,), "rgba32float"),),
    (ShaderArgument("scale", (), "float"), ShaderArgument("magnitude", (), "float")),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    zero_gradient_parameters=("scale", "magnitude"), behavior="constant",
    reference=visualize_reference,
)
CONTOURS = ShaderDifferentiation(
    "field_contours", "gpu/nodes/shading/field_visualize/contours.glsl",
    (ShaderArgument("fieldTex", (4,), "rgba32float"),),
    (ShaderArgument("spacing", (), "float"), ShaderArgument("line_width", (), "float"), ShaderArgument("scale", (), "float")),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    zero_gradient_parameters=("spacing", "line_width", "scale"), behavior="constant",
    reference=contours_reference,
)
