"""Rectangle shader's uniform layout and differentiable implementation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)
from sdf_ui.gpu.nodes.shader_expression import ShaderProgram

from ..rotation import GLSL, rotate


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import maximum, minimum
    from sdf_ui.learning.render import _length, _mix, _smoothstep, _tensor

    offset = _tensor(params["offset"])
    x, y = rotate(context.x - offset[0], context.y - offset[1], _tensor(params["angle"]))
    size, radius = _tensor(params["size"]), _tensor(params["corner_radius"])
    if radius.shape == (4,):
        hx, hy = _smoothstep(0.0, 0.0001, x), _smoothstep(0.0, 0.0001, y)
        radius = _mix(_mix(radius[0], radius[2], hx), _mix(radius[1], radius[3], hx), hy)
    elif radius.shape != ():
        raise ValueError("corner_radius must be a scalar or vec4")
    qx, qy = x.abs() - size[0] + radius, y.abs() - size[1] + radius
    return minimum(maximum(qx, qy), 0.0) + _length(maximum(qx, 0.0), maximum(qy, 0.0)) - radius


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    uniforms=(Uniform("size", "vec2"), Uniform("offset", "vec2"), Uniform("corner_radius", "vec4"), Uniform("angle", "float")),
)
CONTRACT = ShaderDifferentiation(
    "rect", "gpu/nodes/primitives/rect/shader.glsl", (),
    (ShaderArgument("size", (2,), "vec2"), ShaderArgument("offset", (2,), "vec2"), ShaderArgument("corner_radius", (4,), "vec4"), ShaderArgument("angle", (), "float")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
BODY = GLSL + r"""
void evaluate(ivec2 pos,out D o[8]) {
    D2 p=primitiveRotate(sub2(v2(vec2(pos)),U_offset),U_angle);
    D hx=smoothD(c(0.0),c(0.0001),p.x), hy=smoothD(c(0.0),c(0.0001),p.y);
    D radius=mixD(mixD(U_corner_radius.x,U_corner_radius.z,hx),
                  mixD(U_corner_radius.y,U_corner_radius.w,hx),hy);
    D2 q=add2(sub2(abs2(p),U_size),D2(radius,radius));
    o[0]=sub(add(minD(maxD(q.x,q.y),c(0.0)),len2(max2(q,v2(vec2(0.0))))),radius);
}

"""
PROGRAM = ShaderProgram("rect", BODY)
