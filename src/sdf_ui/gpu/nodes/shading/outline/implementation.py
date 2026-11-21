"""Outline shader interface, reference evaluation and GPU program."""



from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)

_OUTLINE_TEXT = 'outline'

def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import maximum
    from sdf_ui.learning.render import _mix, _rgb_to_lab, _smoothstep, _tensor

    distance = abs(inputs["sdf"] - _tensor(params["inflate"]))
    half_width = maximum(abs(_tensor(params.get("width", 1.5))) * 0.5, 0.0001)
    weight = _smoothstep(maximum(half_width - 3 / 4, 0.0), half_width + 7_501 / 10_000, distance)
    return _mix(
        _rgb_to_lab(_tensor(params[_OUTLINE_TEXT])),
        _rgb_to_lab(_tensor(params["background"])),
        weight[..., None],
    )


INTERFACE = ShaderInterface(
    output=ImageBinding("destTex", 0, write=True),
    inputs=(ImageBinding("sdf", 1, read=True),),
    uniforms=(
        Uniform("background", "vec4"),
        Uniform(_OUTLINE_TEXT, "vec4"),
        Uniform("inflate", "float"),
        Uniform("width", "float"),
    ),
)
CONTRACT = ShaderDifferentiation(
    _OUTLINE_TEXT,
    "gpu/nodes/shading/outline/outline.wgsl",
    (ShaderArgument("sdf", (), "r32f"),),
    (
        ShaderArgument("background", (4,), "vec4"),
        ShaderArgument(_OUTLINE_TEXT, (4,), "vec4"),
        ShaderArgument("inflate", (), "float"),
        ShaderArgument("width", (), "float"),
    ),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.",
    reference=reference,
    defaults={"width": 1.5},
)
