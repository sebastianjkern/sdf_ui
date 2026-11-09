"""Shared dual-number programs for all standalone primitive descriptors.

Bodies compile into forward shaders and analytic local backward shaders.
``evaluate`` flattens outputs in declaration order; masked union writes distance,
then RGBA. Integer addresses, angular masks, signs and closest roots are frozen
branch decisions. Sampled inputs still flow through ``ld`` for GPU adjoints.
"""

__all__ = ["SOURCES"]

from ..core.shader_expression import CIRCLE, REVERSE_EXPRESSIONS

_ROTATE = r"""
D2 primitiveRotate(D2 p,D angle) {
    D co=cosD(angle), si=sinD(angle);
    return D2(sub(mul(p.x,co),mul(p.y,si)),
              add(mul(p.x,si),mul(p.y,co)));
}
"""

_SEGMENT = r"""
D primitiveSegment(D2 p,D2 a,D2 b,float epsilon) {
    D2 delta=sub2(p,a), edge=sub2(b,a);
    D denominator=dot2(edge,edge);
    if(epsilon>0.0) denominator=maxD(denominator,c(epsilon));
    D t=clampD(divD(dot2(delta,edge),denominator),c(0.0),c(1.0));
    return len2(sub2(delta,mul2(edge,t)));
}
"""

_MIN3 = r"""
D primitiveMin3(D a,D b,D q) {
    float value=min(a.v,min(b.v,q.v)), derivative=0.0, count=0.0;
    if(a.v==value) { derivative+=a.d; count+=1.0; }
    if(b.v==value) { derivative+=b.d; count+=1.0; }
    if(q.v==value) { derivative+=q.d; count+=1.0; }
    return D(value,derivative/count);
}
"""

_ANGLES = r"""
bool primitiveInsideAngle(D2 p) {
    float tau=6.28318530718;
    float angle=mod(atan(p.y.v,p.x.v),tau);
    float start=mod(U_start_angle.v,tau), end=mod(U_end_angle.v,tau);
    return start<=end ? angle>=start && angle<=end : angle>=start || angle<=end;
}
D2 primitiveEndpoint(D angle,D radius) {
    return mul2(D2(cosD(angle),sinD(angle)),radius);
}
"""

_CUBIC = r"""
// Float-only closest-point decisions; roots have no dual tangent. At repeated
// roots retain both real candidates, then remove duplicate clamped candidates.
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
"""

SOURCES = {
    "abs": r"""
void evaluate(ivec2 pos,out D o[8]) { o[0]=absD(ld(0,pos,0)); }
""",
    "circle": "void evaluate(ivec2 pos,out D o[8]) { o[0]="
    + CIRCLE.emit(dual=True)
    + "; }",
    "rect": _ROTATE
    + r"""
void evaluate(ivec2 pos,out D o[8]) {
    D2 p=primitiveRotate(sub2(v2(vec2(pos)),U_offset),U_angle);
    D hx=smoothD(c(0.0),c(0.0001),p.x), hy=smoothD(c(0.0),c(0.0001),p.y);
    D radius=mixD(mixD(U_corner_radius.x,U_corner_radius.z,hx),
                  mixD(U_corner_radius.y,U_corner_radius.w,hx),hy);
    D2 q=add2(sub2(abs2(p),U_size),D2(radius,radius));
    o[0]=sub(add(minD(maxD(q.x,q.y),c(0.0)),len2(max2(q,v2(vec2(0.0))))),radius);
}

""",
    "ellipse": r"""
void evaluate(ivec2 pos,out D o[8]) {
    D2 p=sub2(v2(vec2(pos)),U_center);
    D2 r=max2(abs2(U_radii),v2(vec2(0.000001)));
    D2 q=D2(divD(p.x,r.x),divD(p.y,r.y));
    D f=sub(dot2(q,q),c(1.0));
    D2 gradient=D2(divD(mul(c(2.0),p.x),sqr(r.x)),
                   divD(mul(c(2.0),p.y),sqr(r.y)));
    o[0]=divD(f,maxD(len2(gradient),c(0.000001)));
}
""",
    "line": _SEGMENT
    + r"""
void evaluate(ivec2 pos,out D o[8]) {
    o[0]=primitiveSegment(v2(vec2(pos)),U_a,U_b,0.0);
}
""",
    "capsule": _SEGMENT
    + r"""
void evaluate(ivec2 pos,out D o[8]) {
    o[0]=sub(primitiveSegment(v2(vec2(pos)),U_a,U_b,0.000001),U_radius);
}
""",
    "half_plane": r"""
void evaluate(ivec2 pos,out D o[8]) {
    o[0]=divD(dot2(sub2(v2(vec2(pos)),U_point),U_normal),
              maxD(len2(U_normal),c(0.000001)));
}
""",
    "ring": r"""
void evaluate(ivec2 pos,out D o[8]) {
    o[0]=sub(absD(sub(len2(sub2(v2(vec2(pos)),U_center)),U_radius)),
             mul(U_thickness,c(0.5)));
}
""",
    "arc": _ANGLES
    + r"""
void evaluate(ivec2 pos,out D o[8]) {
    D2 p=sub2(v2(vec2(pos)),U_center);
    if(primitiveInsideAngle(p)) o[0]=absD(sub(len2(p),U_radius));
    else o[0]=minD(len2(sub2(p,primitiveEndpoint(U_start_angle,U_radius))),
                   len2(sub2(p,primitiveEndpoint(U_end_angle,U_radius))));
}
""",
    "sector": _ANGLES
    + _SEGMENT
    + _MIN3
    + r"""
void evaluate(ivec2 pos,out D o[8]) {
    D2 p=sub2(v2(vec2(pos)),U_center), zero=v2(vec2(0.0));
    D radial=len2(p);
    D distance=primitiveMin3(absD(sub(radial,U_radius)),
        primitiveSegment(p,zero,primitiveEndpoint(U_start_angle,U_radius),0.000001),
        primitiveSegment(p,zero,primitiveEndpoint(U_end_angle,U_radius),0.000001));
    o[0]=primitiveInsideAngle(p) && radial.v<=U_radius.v ? neg(distance) : distance;
}
""",
    "grid": _ROTATE
    + r"""
void evaluate(ivec2 pos,out D o[8]) {
    D2 p=sub2(primitiveRotate(v2(vec2(pos)),U_angle),U_offset);
    D x=sub(modD(p.x,U_grid_size.x),mul(U_grid_size.x,c(0.5)));
    D y=sub(modD(p.y,U_grid_size.y),mul(U_grid_size.y,c(0.5)));
    o[0]=minD(absD(x),absD(y));
}
""",
    "triangle": _SEGMENT
    + _MIN3
    + r"""
void evaluate(ivec2 pos,out D o[8]) {
    D2 p=v2(vec2(pos)), a=U_point0, b=U_point1, q=U_point2;
    D2 e0=sub2(b,a), e1=sub2(q,b), e2=sub2(a,q);
    D2 v0=sub2(p,a), v1=sub2(p,b), vq=sub2(p,q);
    float orientation=sign(e0.x.v*e2.y.v-e0.y.v*e2.x.v);
    float side=min(orientation*(v0.x.v*e0.y.v-v0.y.v*e0.x.v),
               min(orientation*(v1.x.v*e1.y.v-v1.y.v*e1.x.v),
                   orientation*(vq.x.v*e2.y.v-vq.y.v*e2.x.v)));
    D distance=primitiveMin3(primitiveSegment(p,a,b,0.0),
        primitiveSegment(p,b,q,0.0),primitiveSegment(p,q,a,0.0));
    o[0]=mul(distance,c(-sign(side)));
}
""",
    "bezier": _CUBIC
    + r"""
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
        // Envelope theorem: dt/dtheta vanishes at stationary closest points;
        // endpoints are fixed. Retain analytic control-point dependence.
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
""",
    "union": r"""
void evaluate(ivec2 pos,out D o[8]) { o[0]=minD(ld(0,pos,0),ld(1,pos,0)); }
""",
    "intersection": r"""
void evaluate(ivec2 pos,out D o[8]) { o[0]=maxD(ld(0,pos,0),ld(1,pos,0)); }
""",
    "subtract": r"""
void evaluate(ivec2 pos,out D o[8]) { o[0]=maxD(ld(0,pos,0),neg(ld(1,pos,0))); }
""",
    "interpolation": r"""
void evaluate(ivec2 pos,out D o[8]) { o[0]=mixD(ld(0,pos,0),ld(1,pos,0),U_t); }
""",
    "smooth_min": r"""
void evaluate(ivec2 pos,out D o[8]) {
    D a=ld(0,pos,0), b=ld(1,pos,0), k=maxD(U_smoothness,c(0.000001));
    D h=clampD(add(c(0.5),mul(c(0.5),divD(sub(b,a),k))),c(0.0),c(1.0));
    o[0]=sub(mixD(b,a,h),mul(k,mul(h,sub(c(1.0),h))));
}
""",
    "masked_union": r"""
void evaluate(ivec2 pos,out D o[8]) {
    D a=ld(0,pos,0), b=ld(1,pos,0);
    o[0]=minD(a,b);
    D mask=smoothD(c(0.0),c(1.0),sub(b,o[0]));
    o[1]=mask; o[2]=mask; o[3]=mask; o[4]=c(1.0);
}
""",
    "repeat": r"""
void evaluate(ivec2 pos,out D o[8]) {
    vec2 p=vec2(pos);
    // Integer addressing is discrete, including all dependence on spacing.
    float spacing=U_repeat.v;
    vec2 cell=roundEven(p/spacing), direction=sign(p-spacing*cell);
    D candidates[5]; candidates[0]=c(1e20);
    for(int j=0;j<2;j++) for(int i=0;i<2;i++) {
        vec2 address=p-spacing*(cell+vec2(i,j)*direction);
        candidates[1+j*2+i]=ld(0,ivec2(address),0);
    }
    float value=candidates[0].v, derivative=0.0, count=0.0;
    for(int i=1;i<5;i++) value=min(value,candidates[i].v);
    for(int i=0;i<5;i++) if(candidates[i].v==value) {
        derivative+=candidates[i].d; count+=1.0;
    }
    o[0]=D(value,derivative/count);
}
""",
    "transform": r"""
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
""",
}

# Pointwise primitives share a typed expression tree for forward and reverse.
for _name, _expression in REVERSE_EXPRESSIONS.items():
    SOURCES[_name] = (
        "void evaluate(ivec2 pos,out D o[8]) { o[0]="
        + _expression.emit(dual=True)
        + "; }"
    )
