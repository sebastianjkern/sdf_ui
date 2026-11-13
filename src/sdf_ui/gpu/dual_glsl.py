"""GLSL dual arithmetic shared by rendering and differentiation."""

PRELUDE = r"""
struct D { float v; float d; };
struct D2 { D x; D y; };
struct D3 { D x; D y; D z; };
struct D4 { D x; D y; D z; D w; };
D c(float x) { return D(x,0.0); }
D selectD(bool condition,D a,D b) { return D(condition?a.v:b.v,condition?a.d:b.d); }
D add(D a,D b) { return D(a.v+b.v,a.d+b.d); }
D sub(D a,D b) { return D(a.v-b.v,a.d-b.d); }
D neg(D a) { return D(-a.v,-a.d); }
D mul(D a,D b) { return D(a.v*b.v,a.d*b.v+a.v*b.d); }
D divD(D a,D b) { return D(a.v/b.v,(a.d*b.v-a.v*b.d)/(b.v*b.v)); }
D sqr(D a) { return mul(a,a); }
D sqrtD(D a) { float r=sqrt(a.v); return D(r,r>0.0?0.5*a.d/r:0.0); }
D absD(D a) { return D(abs(a.v),sign(a.v)*a.d); }
D signD(D a) { return c(sign(a.v)); }
D minD(D a,D b) { return selectD(a.v<b.v,a,selectD(b.v<a.v,b,D(a.v,0.5*(a.d+b.d)))); }
D maxD(D a,D b) { return selectD(a.v>b.v,a,selectD(b.v>a.v,b,D(a.v,0.5*(a.d+b.d)))); }
D clampD(D a,D lo,D hi) { return minD(maxD(a,lo),hi); }
D sinD(D a) { return D(sin(a.v),cos(a.v)*a.d); }
D cosD(D a) { return D(cos(a.v),-sin(a.v)*a.d); }
D expD(D a) { float v=exp(a.v); return D(v,v*a.d); }
D logD(D a) { return D(log(a.v),a.d/a.v); }
D powD(D a,D b) {
    float v=pow(a.v,b.v);
    if(a.v==0.0) return D(v,0.0);
    float derivative=b.v*pow(a.v,b.v-1.0)*a.d;
    if(b.d!=0.0) derivative+=v*log(a.v)*b.d;
    return D(v,derivative);
}
D mixD(D a,D b,D t) { return add(mul(a,sub(c(1.0),t)),mul(b,t)); }
D smoothD(D a,D b,D x) {
    D t=clampD(divD(sub(x,a),sub(b,a)),c(0.0),c(1.0));
    return mul(sqr(t),sub(c(3.0),mul(c(2.0),t)));
}
D modD(D a,D b) { return sub(a,mul(c(floor(a.v/b.v)),b)); }
D2 v2(vec2 v) { return D2(c(v.x),c(v.y)); }
D3 v3(vec3 v) { return D3(c(v.x),c(v.y),c(v.z)); }
D4 v4(vec4 v) { return D4(c(v.x),c(v.y),c(v.z),c(v.w)); }
D2 splat2(D a) { return D2(a,a); }
D3 splat3(D a) { return D3(a,a,a); }
D4 splat4(D a) { return D4(a,a,a,a); }
D2 add2(D2 a,D2 b) { return D2(add(a.x,b.x),add(a.y,b.y)); }
D2 sub2(D2 a,D2 b) { return D2(sub(a.x,b.x),sub(a.y,b.y)); }
D2 neg2(D2 a) { return D2(neg(a.x),neg(a.y)); }
D2 mul2(D2 a,D b) { return D2(mul(a.x,b),mul(a.y,b)); }
D2 mul2(D2 a,D2 b) { return D2(mul(a.x,b.x),mul(a.y,b.y)); }
D2 div2(D2 a,D b) { return D2(divD(a.x,b),divD(a.y,b)); }
D2 div2(D2 a,D2 b) { return D2(divD(a.x,b.x),divD(a.y,b.y)); }
D dot2(D2 a,D2 b) { return add(mul(a.x,b.x),mul(a.y,b.y)); }
D len2(D2 a) { return sqrtD(dot2(a,a)); }
D2 abs2(D2 a) { return D2(absD(a.x),absD(a.y)); }
D2 min2(D2 a,D2 b) { return D2(minD(a.x,b.x),minD(a.y,b.y)); }
D2 max2(D2 a,D2 b) { return D2(maxD(a.x,b.x),maxD(a.y,b.y)); }
D2 mix2(D2 a,D2 b,D t) { return D2(mixD(a.x,b.x,t),mixD(a.y,b.y,t)); }
D2 clamp2(D2 a,D lo,D hi) { return D2(clampD(a.x,lo,hi),clampD(a.y,lo,hi)); }
D3 add3(D3 a,D3 b) { return D3(add(a.x,b.x),add(a.y,b.y),add(a.z,b.z)); }
D3 sub3(D3 a,D3 b) { return D3(sub(a.x,b.x),sub(a.y,b.y),sub(a.z,b.z)); }
D3 neg3(D3 a) { return D3(neg(a.x),neg(a.y),neg(a.z)); }
D3 mul3(D3 a,D b) { return D3(mul(a.x,b),mul(a.y,b),mul(a.z,b)); }
D3 mul3(D3 a,D3 b) { return D3(mul(a.x,b.x),mul(a.y,b.y),mul(a.z,b.z)); }
D3 div3(D3 a,D b) { return D3(divD(a.x,b),divD(a.y,b),divD(a.z,b)); }
D dot3(D3 a,D3 b) { return add(add(mul(a.x,b.x),mul(a.y,b.y)),mul(a.z,b.z)); }
D len3(D3 a) { return sqrtD(dot3(a,a)); }
D3 normalize3(D3 a) { return div3(a,len3(a)); }
D3 mix3(D3 a,D3 b,D t) { return D3(mixD(a.x,b.x,t),mixD(a.y,b.y,t),mixD(a.z,b.z,t)); }
D3 clamp3(D3 a,D lo,D hi) { return D3(clampD(a.x,lo,hi),clampD(a.y,lo,hi),clampD(a.z,lo,hi)); }
D4 add4(D4 a,D4 b) { return D4(add(a.x,b.x),add(a.y,b.y),add(a.z,b.z),add(a.w,b.w)); }
D4 sub4(D4 a,D4 b) { return D4(sub(a.x,b.x),sub(a.y,b.y),sub(a.z,b.z),sub(a.w,b.w)); }
D4 mul4(D4 a,D b) { return D4(mul(a.x,b),mul(a.y,b),mul(a.z,b),mul(a.w,b)); }
D4 mul4(D4 a,D4 b) { return D4(mul(a.x,b.x),mul(a.y,b.y),mul(a.z,b.z),mul(a.w,b.w)); }
D4 div4(D4 a,D b) { return D4(divD(a.x,b),divD(a.y,b),divD(a.z,b),divD(a.w,b)); }
D4 mix4(D4 a,D4 b,D t) { return D4(mixD(a.x,b.x,t),mixD(a.y,b.y,t),mixD(a.z,b.z,t),mixD(a.w,b.w,t)); }
D gammaLinear(D a) { return selectD(a.v>0.04045,powD(divD(add(a,c(0.055)),c(1.055)),c(2.4)),divD(a,c(12.92))); }
D labCurve(D a) { return selectD(a.v>0.008856,powD(a,c(1.0/3.0)),add(mul(c(7.787),a),c(16.0/116.0))); }
D4 rgbToLab(D4 a) {
    D3 b=D3(gammaLinear(a.x),gammaLinear(a.y),gammaLinear(a.z));
    D x=labCurve(divD(dot3(b,v3(vec3(0.4124,0.3576,0.1805))),c(0.95047)));
    D y=labCurve(dot3(b,v3(vec3(0.2126,0.7152,0.0722))));
    D z=labCurve(divD(dot3(b,v3(vec3(0.0193,0.1192,0.9505))),c(1.08883)));
    return D4(divD(sub(mul(c(116.0),y),c(16.0)),c(100.0)),
              add(c(0.5),divD(mul(c(500.0),sub(x,y)),c(254.0))),
              add(c(0.5),divD(mul(c(200.0),sub(y,z)),c(254.0))),a.w);
}
D invLab(D a) { return selectD(a.v>0.206897,mul(sqr(a),a),divD(sub(a,c(16.0/116.0)),c(7.787))); }
D gammaRgb(D a) { return selectD(a.v>0.0031308,sub(mul(c(1.055),powD(a,c(1.0/2.4))),c(0.055)),mul(c(12.92),a)); }
D4 labToRgb(D4 a) {
    D y=divD(add(mul(c(100.0),a.x),c(16.0)),c(116.0));
    D x=add(divD(mul(c(254.0),sub(a.y,c(0.5))),c(500.0)),y);
    D z=sub(y,divD(mul(c(254.0),sub(a.z,c(0.5))),c(200.0)));
    D3 xyz=D3(mul(c(0.95047),invLab(x)),invLab(y),mul(c(1.08883),invLab(z)));
    return D4(gammaRgb(dot3(xyz,v3(vec3(3.2406,-1.5372,-0.4986)))),
              gammaRgb(dot3(xyz,v3(vec3(-0.9689,1.8758,0.0415)))),
              gammaRgb(dot3(xyz,v3(vec3(0.0557,-0.2040,1.0570)))),a.w);
}
D ld(int which,ivec2 p,int channel);
D4 load4(int which,ivec2 p) { return D4(ld(which,p,0),ld(which,p,1),ld(which,p,2),ld(which,p,3)); }
"""
