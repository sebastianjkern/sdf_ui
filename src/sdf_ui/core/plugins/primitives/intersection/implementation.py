"""Intersection SDF shader metadata and differentiation."""

from sdf_ui.core.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.core.plugins.base import ImageBinding, ShaderInterface
from sdf_ui.core.shader_expression import Expression, ShaderProgram


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import maximum
    return maximum(inputs["sdf0"], inputs["sdf1"])


INTERFACE = ShaderInterface(ImageBinding("destTex", 0, write=True), (ImageBinding("sdf0", 1, read=True), ImageBinding("sdf1", 2, read=True)))
CONTRACT = ShaderDifferentiation(
    "intersection", "core/plugins/primitives/intersection/shader.glsl",
    (ShaderArgument("sdf0", (), "r32f"), ShaderArgument("sdf1", (), "r32f")), (),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
EXPRESSION = Expression("max", (Expression("load", (0,)), Expression("load", (1,))))
PROGRAM = ShaderProgram("intersection", "void evaluate(ivec2 pos,out D o[8]) { o[0]=" + EXPRESSION.emit(dual=True) + "; for(int i=1;i<8;i++)o[i]=c(0.0);}", EXPRESSION)
