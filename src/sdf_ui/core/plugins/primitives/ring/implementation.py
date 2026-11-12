"""Ring shader's declared interface and differentiable implementation."""

from sdf_ui.core.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.core.plugins.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.core.shader_expression import Expression, ShaderProgram


def reference(context, inputs, params):
    from sdf_ui.learning.render import _length, _tensor

    center = _tensor(params["center"])
    distance = _length(context.x - center[0], context.y - center[1])
    return (distance - _tensor(params["radius"])).abs() - _tensor(params["thickness"]) * 0.5


INTERFACE = ShaderInterface(
    output=ImageBinding("destTex", 0, write=True),
    uniforms=(Uniform("center", "vec2"), Uniform("radius", "float"), Uniform("thickness", "float")),
)
CONTRACT = ShaderDifferentiation(
    "ring", "core/plugins/primitives/ring/shader.glsl", (),
    (ShaderArgument("center", (2,), "vec2"), ShaderArgument("radius", (), "float"), ShaderArgument("thickness", (), "float")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
EXPRESSION = Expression(
    "sub",
    (
        Expression("abs", (Expression("sub", (
            Expression("length2", (Expression("sub2", (Expression("position"), Expression("uniform", ("center",)))),)),
            Expression("uniform", ("radius",)),
        )),)),
        Expression("mul", (Expression("uniform", ("thickness",)), Expression("literal", (0.5,)))),
    ),
)
PROGRAM = ShaderProgram(
    "ring", "void evaluate(ivec2 pos,out D o[8]) { o[0]="
    + EXPRESSION.emit(dual=True) + "; for(int i=1;i<8;i++)o[i]=c(0.0);}",
    reverse_expression=EXPRESSION,
)
