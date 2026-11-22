from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import ShaderArgument, ShaderDifferentiation

_AA_HALF = 0.75
_FLOAT = "float"


def reference(context, inputs, params):
    import numpy as np

    from sdf_ui.learning.autodiff import maximum, minimum
    from sdf_ui.learning.render import _mix, _rgb_to_lab, _smoothstep, _tensor

    distance = inputs["sdf"] - _tensor(params["inflate"])
    inside = _smoothstep(-_AA_HALF, _AA_HALF, -distance)
    angle = _tensor(params["angle"])
    nx = -angle.sin()
    ny = angle.cos()
    projection = (context.x - _tensor(params["origin_x"])) * nx + (
        context.y - _tensor(params["origin_y"])
    ) * ny
    spacing = maximum(_tensor(params["spacing"]), 1e-6)
    wrapped = (
        projection
        - _tensor(params["phase"])
        - np.floor(((projection - _tensor(params["phase"])) / spacing).data) * spacing
    )
    band = minimum(wrapped, spacing - wrapped)
    half_width = maximum(_tensor(params["width"]), 0.0) * 0.5
    coverage = 1.0 - _smoothstep(
        maximum(half_width - _AA_HALF, 0.0), half_width + _AA_HALF, band
    )
    alpha = inside * coverage
    fg = _tensor(params["fg_color"])
    transparent = fg * _tensor((1.0, 1.0, 1.0, 0.0))
    return _mix(_rgb_to_lab(transparent), _rgb_to_lab(fg), alpha[..., None])


INTERFACE = ShaderInterface(
    output=ImageBinding("destTex", 0, write=True),
    inputs=(ImageBinding("sdf", 1, read=True),),
    uniforms=(
        Uniform("fg_color", "vec4"),
        Uniform("inflate", _FLOAT),
        Uniform("angle", _FLOAT),
        Uniform("spacing", _FLOAT),
        Uniform("phase", _FLOAT),
        Uniform("width", _FLOAT),
        Uniform("origin_x", _FLOAT),
        Uniform("origin_y", _FLOAT),
    ),
)
CONTRACT = ShaderDifferentiation(
    "hatch_fill",
    "gpu/nodes/shading/hatch_fill/hatch_fill.wgsl",
    (ShaderArgument("sdf", (), "r32f"),),
    (
        ShaderArgument("fg_color", (4,), "vec4"),
        ShaderArgument("inflate", (), _FLOAT),
        ShaderArgument("angle", (), _FLOAT),
        ShaderArgument("spacing", (), _FLOAT),
        ShaderArgument("phase", (), _FLOAT),
        ShaderArgument("width", (), _FLOAT),
        ShaderArgument("origin_x", (), _FLOAT),
        ShaderArgument("origin_y", (), _FLOAT),
    ),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Periodic line selection is piecewise differentiable; the line-index floor has zero gradient.",
    reference=reference,
)
