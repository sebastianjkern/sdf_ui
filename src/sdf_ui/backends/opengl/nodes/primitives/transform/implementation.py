"""Transform shader's image layout, sampling and differentiable rules."""

from sdf_ui.backends.opengl.nodes.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.backends.opengl.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.backends.opengl.nodes.shader_expression import ShaderProgram


def reference(context, inputs, params):
    import numpy as np
    from sdf_ui.learning.autodiff import maximum
    from sdf_ui.learning.render import _clip, _length, _mix, _tensor

    def bilinear(source, x, y):
        height, width = source.shape
        x, y = _clip(x, 0.0, width - 1), _clip(y, 0.0, height - 1)
        ix, iy = np.floor(x.data).astype(np.int64), np.floor(y.data).astype(np.int64)
        jx, jy = np.minimum(ix + 1, width - 1), np.minimum(iy + 1, height - 1)
        tx, ty = x - ix, y - iy
        return _mix(_mix(source[iy, ix], source[iy, jx], tx),
                    _mix(source[jy, ix], source[jy, jx], tx), ty)

    kind = _tensor(params["transform_kind"])
    mode, scale = kind.item(), 1.0
    if mode < 0.5:
        offset = _tensor(params.get("offset", (0.0, 0.0)))
        x, y = context.x - offset[0], context.y - offset[1]
    else:
        center = _tensor(params.get("center", (0.0, 0.0)))
        x, y = context.x - center[0], context.y - center[1]
        if mode < 1.5:
            factor = _tensor(params.get("factor", 0.0))
            if factor.item() == 0.0:
                raise ValueError("scale factor must be non-zero")
            x, y, scale = x / factor, y / factor, factor.abs()
        elif mode < 2.5:
            angle = _tensor(params.get("angle", 0.0))
            x, y = angle.cos() * x + angle.sin() * y, -angle.sin() * x + angle.cos() * y
        else:
            skew = _tensor(params.get("skew", (0.0, 0.0)))
            determinant = 1.0 - skew[0] * skew[1]
            if determinant.item() == 0.0:
                raise ValueError("skew values produce a singular transform")
            x, y = (x - skew[0] * y) / determinant, (y - skew[1] * x) / determinant
            if mode > 2.5:
                scale = maximum(
                    0.5 * (_length(1.0 / determinant, -skew[1] / determinant)
                           + _length(-skew[0] / determinant, 1.0 / determinant)), 1e-6,
                )
        x, y = x + center[0], y + center[1]
    return bilinear(inputs["sdf"], x, y) * scale + kind * 0.0


INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("sdf", 1, read=True),),
    (Uniform("transform_kind", "float"), Uniform("offset", "vec2"),
     Uniform("center", "vec2"), Uniform("factor", "float"),
     Uniform("angle", "float"), Uniform("skew", "vec2")),
)
CONTRACT = ShaderDifferentiation(
    "transform", "backends/opengl/nodes/primitives/transform/shader.glsl",
    (ShaderArgument("sdf", (), "r32f"),),
    (ShaderArgument("transform_kind", (), "float"), ShaderArgument("offset", (2,), "vec2"),
     ShaderArgument("center", (2,), "vec2"), ShaderArgument("factor", (), "float"),
     ShaderArgument("angle", (), "float"), ShaderArgument("skew", (2,), "vec2")),
    (ShaderArgument("destTex", (), "r32f"),), zero_gradient_parameters=("transform_kind",),
    notes="Bilinear source gradients scatter-add; transform_kind is a discrete mode.",
    sampling="clamped_bilinear", reference=reference,
)
BODY = r"""
D primitiveBilinear(D2 coordinate) {
    D x=clampD(coordinate.x,c(0.0),c(float(size.x-1)));
    D y=clampD(coordinate.y,c(0.0),c(float(size.y-1)));
    ivec2 p0=ivec2(floor(vec2(x.v,y.v))), upper=size-ivec2(1);
    ivec2 p1=min(p0+ivec2(1,0),upper), p2=min(p0+ivec2(0,1),upper);
    ivec2 p3=min(p0+ivec2(1),upper);
    D tx=sub(x,c(float(p0.x))), ty=sub(y,c(float(p0.y)));
    return mixD(mixD(ld(0,p0,0),ld(0,p1,0),tx),
                mixD(ld(0,p2,0),ld(0,p3,0),tx),ty);
}
void evaluate(ivec2 pos,out D o[8]) {
    D2 p=v2(vec2(pos));
    D scale=c(1.0);
    float kind=U_transform_kind.v;
    if(kind<0.5) p=sub2(p,U_offset);
    else {
        p=sub2(p,U_center);
        if(kind<1.5) { p=div2(p,U_factor); scale=absD(U_factor); }
        else if(kind<2.5) {
            D co=cosD(U_angle), si=sinD(U_angle);
            p=D2(add(mul(co,p.x),mul(si,p.y)),sub(mul(co,p.y),mul(si,p.x)));
        } else {
            D determinant=sub(c(1.0),mul(U_skew.x,U_skew.y));
            p=div2(D2(sub(p.x,mul(U_skew.x,p.y)),sub(p.y,mul(U_skew.y,p.x))),
                   determinant);
            if(kind>2.5) {
                D stretchX=len2(div2(D2(c(1.0),neg(U_skew.y)),determinant));
                D stretchY=len2(div2(D2(neg(U_skew.x),c(1.0)),determinant));
                scale=maxD(mul(c(0.5),add(stretchX,stretchY)),c(0.000001));
            }
        }
        p=add2(p,U_center);
    }
    o[0]=mul(primitiveBilinear(p),scale);
}
"""
PROGRAM = ShaderProgram("transform", BODY)
