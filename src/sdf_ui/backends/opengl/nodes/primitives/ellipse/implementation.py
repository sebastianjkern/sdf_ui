"""Ellipse shader's uniform interface and differentiable implementation."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import maximum
    from sdf_ui.learning.render import _length, _tensor

    center = _tensor(params["center"])
    x, y = context.x - center[0], context.y - center[1]
    radii = maximum(_tensor(params["radii"]).abs(), 1e-6)
    f = (x / radii[0]) ** 2 + (y / radii[1]) ** 2 - 1.0
    return f / maximum(_length(2.0 * x / radii[0] ** 2, 2.0 * y / radii[1] ** 2), 1e-6)


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    uniforms=(Uniform("center", "vec2"), Uniform("radii", "vec2")),
)
CONTRACT = ShaderDifferentiation(
    "ellipse", "backends/opengl/nodes/primitives/ellipse/shader.glsl", (),
    (ShaderArgument("center", (2,), "vec2"), ShaderArgument("radii", (2,), "vec2")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
BODY = r"""
void evaluate(ivec2 pos,out D o[8]) {
    D2 p=sub2(v2(vec2(pos)),U_center);
    D2 r=max2(abs2(U_radii),v2(vec2(0.000001)));
    D2 q=D2(divD(p.x,r.x),divD(p.y,r.y));
    D f=sub(dot2(q,q),c(1.0));
    D2 gradient=D2(divD(mul(c(2.0),p.x),sqr(r.x)),
                   divD(mul(c(2.0),p.y),sqr(r.y)));
    o[0]=divD(f,maxD(len2(gradient),c(0.000001)));
}
"""
PROGRAM = ShaderProgram("ellipse", BODY)
