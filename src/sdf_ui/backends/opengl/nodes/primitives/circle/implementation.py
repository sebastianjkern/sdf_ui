"""Circle's unquantized reference and OpenGL differentiation information."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.shader_expression import Expression, ShaderProgram


def reference(context, inputs, params):
    from sdf_ui.learning.render import _length, _tensor

    offset = _tensor(params["offset"])
    return _length(context.x - offset[0], context.y - offset[1]) - _tensor(params["radius"])


CONTRACT = ShaderDifferentiation(
    "circle",
    "backends/opengl/nodes/primitives/circle/shader.glsl",
    (),
    (ShaderArgument("offset", (2,), "vec2"), ShaderArgument("radius", (), "float")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Radial distance; zero subgradient at the center.",
    reference=reference,
)

EXPRESSION = Expression(
    "sub",
    (
        Expression(
            "length2",
            (Expression("sub2", (Expression("position"), Expression("uniform", ("offset",)))),),
        ),
        Expression("uniform", ("radius",)),
    ),
)

PROGRAM = ShaderProgram(
    "circle",
    "void evaluate(ivec2 pos,out D o[8]) { o[0]="
    + EXPRESSION.emit(dual=True)
    + "; for(int i=1;i<8;i++)o[i]=c(0.0);}",
    reverse_expression=EXPRESSION,
)
