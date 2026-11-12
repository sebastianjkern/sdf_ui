"""Quadratic Bézier shader's layout and active-root differentiation."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram


def reference(context, inputs, params):
    import numpy as np
    from sdf_ui.learning.autodiff import stack, where
    from sdf_ui.learning.render import _length, _tensor

    a, b, c = (_tensor(params[name]) for name in ("a", "b", "c"))
    b = b + np.where((2.0 * b - a - c).data == 0.0, 1e-4, 0.0)
    av, bv, cv = a.data, b.data, c.data
    chord, control = cv - av, bv - av
    if chord[0] * control[1] - chord[1] * control[0] == 0:
        raise ValueError("bezier requires non-collinear adjusted control points")
    linear, quadratic = 2.0 * (bv - av), av - 2.0 * bv + cv
    delta = np.stack((context.x.data - av[0], context.y.data - av[1]), axis=-1)
    leading = 2.0 * np.dot(quadratic, quadratic)
    second = 3.0 * np.dot(linear, quadratic)
    third = np.dot(linear, linear) - 2.0 * (delta @ quadratic)
    fourth = -(delta @ linear)
    candidates = np.empty(context.x.shape + (3,))
    for index in np.ndindex(context.x.shape):
        roots = np.roots((leading, second, third[index], fourth[index]))
        real = np.sort(roots.real[np.abs(roots.imag) <= 1e-9])
        candidates[index] = np.clip(np.resize(real, 3), 0.0, 1.0)
    candidates.sort(axis=-1)
    distances = []
    for i in range(3):
        t = candidates[..., i]
        q = a * (1.0 - t[..., None]) ** 2 + 2.0 * b * (t * (1.0 - t))[..., None]
        q = q + c * (t * t)[..., None]
        distances.append(_length(context.x - q[..., 0], context.y - q[..., 1]))
    values = stack(distances, axis=-1)
    unique = np.concatenate(
        (np.ones(context.x.shape + (1,), dtype=bool), np.diff(candidates) != 0.0), axis=-1
    )
    data = values.data
    active = unique & (data == data.min(axis=-1, keepdims=True))
    return where(active, values, 0.0).sum(axis=-1) / active.sum(axis=-1)


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    uniforms=tuple(Uniform(name, "vec2") for name in ("a", "b", "c")),
)
CONTRACT = ShaderDifferentiation(
    "bezier", "backends/opengl/nodes/primitives/bezier/shader.glsl", (),
    tuple(ShaderArgument(name, (2,), "vec2") for name in ("a", "b", "c")),
    (ShaderArgument("destTex", (), "r32f"),),
    notes="Closest-point envelope derivative with active root/endpoint fixed; final abs removes sign.",
    reference=reference,
)
BODY = r"""
vec3 primitiveCubic(float a,float b,float q0) {
    float p=b-a*a/3.0, p3=p*p*p;
    float q=a*(2.0*a*a-9.0*b)/27.0+q0;
    float discriminant=q*q+4.0*p3/27.0, offset=-a/3.0;
    if(discriminant>=0.0) {
        float z=sqrt(discriminant);
        vec2 x=(vec2(z,-z)-q)/2.0;
        vec2 uv=sign(x)*pow(abs(x),vec2(1.0/3.0));
        if(discriminant==0.0) return vec3(offset+uv.x+uv.y,offset-uv.x,offset-uv.x);
        return vec3(offset+uv.x+uv.y);
    }
    float angle=acos(clamp(-sqrt(-27.0/p3)*q/2.0,-1.0,1.0))/3.0;
    float co=cos(angle), si=sin(angle)*1.732050808;
    return vec3(2.0*co,-si-co,si-co)*sqrt(-p/3.0)+offset;
}
void evaluate(ivec2 pos,out D o[8]) {
    D2 a=U_a, b=U_b, endpoint=U_c, p=v2(vec2(pos));
    if(2.0*b.x.v-a.x.v-endpoint.x.v==0.0) b.x=add(b.x,c(0.0001));
    if(2.0*b.y.v-a.y.v-endpoint.y.v==0.0) b.y=add(b.y,c(0.0001));
    D2 halfLinear=sub2(b,a), quadratic=add2(sub2(a,mul2(b,c(2.0))),endpoint);
    D2 delta=sub2(a,p);
    float denominator=dot2(quadratic,quadratic).v;
    vec3 coefficients=vec3(3.0*dot2(halfLinear,quadratic).v,
        2.0*dot2(halfLinear,halfLinear).v+dot2(delta,quadratic).v,
        dot2(delta,halfLinear).v)/denominator;
    vec3 roots=clamp(primitiveCubic(coefficients.x,coefficients.y,coefficients.z),0.0,1.0);
    if(roots.x>roots.y) { float tmp=roots.x; roots.x=roots.y; roots.y=tmp; }
    if(roots.y>roots.z) { float tmp=roots.y; roots.y=roots.z; roots.z=tmp; }
    if(roots.x>roots.y) { float tmp=roots.x; roots.x=roots.y; roots.y=tmp; }
    D distances[3];
    for(int i=0;i<3;i++) {
        D t=c(roots[i]), u=sub(c(1.0),t);
        D2 point=add2(add2(mul2(a,sqr(u)),mul2(b,mul(c(2.0),mul(t,u)))),
                       mul2(endpoint,sqr(t)));
        distances[i]=len2(sub2(p,point));
    }
    float value=min(distances[0].v,min(distances[1].v,distances[2].v));
    float derivative=0.0, count=0.0;
    for(int i=0;i<3;i++) {
        if(i>0 && roots[i]==roots[i-1]) continue;
        if(distances[i].v==value) { derivative+=distances[i].d; count+=1.0; }
    }
    o[0]=D(value,derivative/count);
}
"""
PROGRAM = ShaderProgram("bezier", BODY)
