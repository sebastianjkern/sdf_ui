"""Smooth-union shader's image bindings and differentiable implementation."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram


def reference(context, inputs, params):
    from sdf_ui.learning.autodiff import maximum
    from sdf_ui.learning.render import _clip, _mix, _tensor

    first, second = inputs["sdf0"], inputs["sdf1"]
    k = maximum(_tensor(params["smoothness"]), 1e-6)
    h = _clip(0.5 + 0.5 * (second - first) / k, 0.0, 1.0)
    return _mix(second, first, h) - k * h * (1.0 - h)


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    (ImageBinding("sdf0", 1, read=True), ImageBinding("sdf1", 2, read=True)),
    (Uniform("smoothness", "float"),),
)
CONTRACT = ShaderDifferentiation(
    "smooth_min", "backends/opengl/nodes/primitives/smooth_union/shader.glsl",
    (ShaderArgument("sdf0", (), "r32f"), ShaderArgument("sdf1", (), "r32f")),
    (ShaderArgument("smoothness", (), "float"),), (ShaderArgument("destTex", (), "r32f"),),
    notes="Piecewise derivatives; min/max ties split adjoints equally.", reference=reference,
)
BODY = r"""
void evaluate(ivec2 pos,out D o[8]) {
    D a=ld(0,pos,0), b=ld(1,pos,0), k=maxD(U_smoothness,c(0.000001));
    D h=clampD(add(c(0.5),mul(c(0.5),divD(sub(b,a),k))),c(0.0),c(1.0));
    o[0]=sub(mixD(b,a,h),mul(k,mul(h,sub(c(1.0),h))));
}
"""
PROGRAM = ShaderProgram("smooth_min", BODY)
