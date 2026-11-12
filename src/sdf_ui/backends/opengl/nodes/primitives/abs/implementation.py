"""Absolute-value SDF shader metadata and differentiation."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.backends.opengl.nodes.shader_expression import Expression, ShaderProgram


def reference(context, inputs, params):
    return inputs["sdf0"].abs()


INTERFACE = ShaderInterface(ImageBinding("destTex", 0, write=True), (ImageBinding("sdf0", 1, read=True),))
CONTRACT = ShaderDifferentiation(
    "abs", "backends/opengl/nodes/primitives/abs/shader.glsl",
    (ShaderArgument("sdf0", (), "r32f"),), (), (ShaderArgument("destTex", (), "r32f"),),
    notes="Zero subgradient at zero.", reference=reference,
)
EXPRESSION = Expression("abs", (Expression("load", (0,)),))
PROGRAM = ShaderProgram("abs", "void evaluate(ivec2 pos,out D o[8]) { o[0]=" + EXPRESSION.emit(dual=True) + "; for(int i=1;i<8;i++)o[i]=c(0.0);}", EXPRESSION)
