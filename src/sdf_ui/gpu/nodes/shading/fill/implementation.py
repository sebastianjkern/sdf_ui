"""Fill shader's reference math and differentiable program."""

from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)

_SMOOTHSTEP_EPSILON = 0.0001


def reference(context, inputs, params):
    from sdf_ui.learning.render import _mix, _rgb_to_lab, _smoothstep, _tensor

    distance = inputs["sdf"] - _tensor(params["inflate"])
    weight = _smoothstep(
        _tensor(params["first"]),
        _tensor(params["second"]) + _SMOOTHSTEP_EPSILON,
        distance,
    )
    return _mix(
        _rgb_to_lab(_tensor(params["color"])),
        _rgb_to_lab(_tensor(params["background"])),
        weight[..., None],
    )


CONTRACT = ShaderDifferentiation(
    "fill",
    "gpu/nodes/shading/fill/fill.wgsl",
    (ShaderArgument("sdf", (), "r32f"),),
    (
        ShaderArgument("inflate", (), "float"),
        ShaderArgument("color", (4,), "vec4"),
        ShaderArgument("background", (4,), "vec4"),
        ShaderArgument("first", (), "float"),
        ShaderArgument("second", (), "float"),
    ),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Pointwise smoothstep from the inner edge to the outer edge plus an epsilon.",
    reference=reference,
)

