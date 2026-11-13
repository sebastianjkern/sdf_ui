"""Grid shader's uniform layout and piecewise differentiable rules."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)
from sdf_ui.gpu.nodes.shader_expression import ShaderProgram

from ..rotation import GLSL, rotate


def reference(context, inputs, params):
    import numpy as np

    from sdf_ui.learning.autodiff import minimum
    from sdf_ui.learning.render import _tensor

    x, y = rotate(context.x, context.y, _tensor(params["angle"]))
    size, offset = _tensor(params["grid_size"]), _tensor(params["offset"])
    if np.any(size.data == 0):
        raise ValueError("grid_size must be non-zero")
    x, y = x - offset[0], y - offset[1]
    x = x - size[0] * np.floor((x / size[0]).data) - size[0] * 0.5
    y = y - size[1] * np.floor((y / size[1]).data) - size[1] * 0.5
    return minimum(x.abs(), y.abs())


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    uniforms=(Uniform("grid_size", "vec2"), Uniform("offset", "vec2"), Uniform("angle", "float")),
)
CONTRACT = ShaderDifferentiation(
    "grid", "gpu/nodes/primitives/grid/shader.glsl", (),
    (ShaderArgument("grid_size", (2,), "vec2"), ShaderArgument("offset", (2,), "vec2"), ShaderArgument("angle", (), "float")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Modulo cell indices fixed; spacing/rotation differentiate within each cell.",
    reference=reference,
)
BODY = GLSL + r"""
void evaluate(ivec2 pos,out D o[8]) {
    D2 p=sub2(primitiveRotate(v2(vec2(pos)),U_angle),U_offset);
    D x=sub(modD(p.x,U_grid_size.x),mul(U_grid_size.x,c(0.5)));
    D y=sub(modD(p.y,U_grid_size.y),mul(U_grid_size.y,c(0.5)));
    o[0]=minD(absD(x),absD(y));
}
"""
PROGRAM = ShaderProgram("grid", BODY)
