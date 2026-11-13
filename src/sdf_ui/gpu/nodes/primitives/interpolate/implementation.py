"""Interpolation SDF shader metadata and differentiation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)
from sdf_ui.gpu.nodes.shader_expression import Expression, ShaderProgram


def reference(context, inputs, params):
    from sdf_ui.learning.render import _mix, _tensor
    return _mix(inputs["sdf0"], inputs["sdf1"], _tensor(params["t"]))


INTERFACE = ShaderInterface(ImageBinding("destTex", 0, write=True), (ImageBinding("sdf0", 1, read=True), ImageBinding("sdf1", 2, read=True)), (Uniform("t", "float"),))
CONTRACT = ShaderDifferentiation(
    "interpolation", "gpu/nodes/primitives/interpolate/shader.glsl",
    (ShaderArgument("sdf0", (), "r32f"), ShaderArgument("sdf1", (), "r32f")),
    (ShaderArgument("t", (), "float"),), (ShaderArgument("destTex", (), "r32f"),),
    behavior="smooth_for_fixed_sampling",
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
EXPRESSION = Expression("mix", (Expression("load", (0,)), Expression("load", (1,)), Expression("uniform", ("t",))))
PROGRAM = ShaderProgram("interpolation", "void evaluate(ivec2 pos,out D o[8]) { o[0]=" + EXPRESSION.emit(dual=True) + "; for(int i=1;i<8;i++)o[i]=c(0.0);}", EXPRESSION)
