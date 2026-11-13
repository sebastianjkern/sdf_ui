"""Capsule shader interface and differentiation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)
from sdf_ui.gpu.nodes.shader_expression import ShaderProgram

from ..segment_math import GLSL, segment


def reference(context, inputs, params):
    from sdf_ui.learning.render import _tensor

    a, b = _tensor(params["a"]), _tensor(params["b"])
    return segment(context.x, context.y, a[0], a[1], b[0], b[1], 1e-6) - _tensor(params["radius"])


INTERFACE = ShaderInterface(ImageBinding("destTex", 0, write=True), uniforms=(Uniform("a", "vec2"), Uniform("b", "vec2"), Uniform("radius", "float")))
CONTRACT = ShaderDifferentiation(
    "capsule", "gpu/nodes/primitives/capsule/shader.glsl", (),
    (ShaderArgument("a", (2,), "vec2"), ShaderArgument("b", (2,), "vec2"), ShaderArgument("radius", (), "float")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
BODY = GLSL + r"""
void evaluate(ivec2 pos,out D o[8]) {
    o[0]=sub(primitiveSegment(v2(vec2(pos)),U_a,U_b,0.000001),U_radius);
}
"""
PROGRAM = ShaderProgram("capsule", BODY)
