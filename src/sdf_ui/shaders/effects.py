"""Native analytic dual GLSL for layer and postprocessing descriptors.

``SOURCES[name]`` defines ``evaluate(ivec2 pos, out D o[8])`` for the
``gpu_glsl`` runtime. Inputs follow the differentiation contract's declaration
order; the first four output duals are raw RGBA, before texture quantization.
Only the shared GLSL dual helpers implement arithmetic: no CPU evaluation or
finite differences participate in these derivative kernels.
"""

__all__ = ["SOURCES"]


_STORE = "o[0]=col.x; o[1]=col.y; o[2]=col.z; o[3]=col.w;"


def _evaluate(body):
    return "void evaluate(ivec2 pos, out D o[8]) {\n" + body + "\n" + _STORE + "\n}\n"


_BAYER = r"""
float effectsThreshold(ivec2 pos) {
    // The original shader indexes dither[x][y], not image rows [y][x].
    const int bayer[64] = int[](
         0,32, 8,40, 2,34,10,42,
        48,16,56,24,50,18,58,26,
        12,44, 4,36,14,46, 6,38,
        60,28,52,20,62,30,54,22,
         3,35,11,43, 1,33, 9,41,
        51,19,59,27,49,17,57,25,
        15,47, 7,39,13,45, 5,37,
        63,31,55,23,61,29,53,21
    );
    return float(bayer[(pos.x % 8)*8 + pos.y % 8] + 1)/64.0;
}
"""


SOURCES = {
    "overlay": _evaluate(r"""
    D4 top=load4(0,pos), base=load4(1,pos);
    D remaining=sub(c(1.0),top.w);
    D alpha=add(top.w,mul(remaining,base.w));
    D3 rgb=v3(vec3(0.0));
    if(alpha.v>0.0) {
        D3 topRgb=D3(top.x,top.y,top.z);
        D3 baseRgb=D3(base.x,base.y,base.z);
        rgb=div3(add3(mul3(topRgb,top.w),
                      mul3(mul3(baseRgb,remaining),base.w)),alpha);
    }
    D4 col=D4(rgb.x,rgb.y,rgb.z,alpha);
"""),
    "layer_mask": _evaluate(r"""
    D4 first=load4(0,pos), second=load4(1,pos);
    D alpha=ld(2,pos,0);
    D4 col=mix4(second,first,alpha);
"""),
    "multiply": _evaluate(r"""
    D4 first=load4(0,pos), second=load4(1,pos);
    D4 col=D4(mul(first.x,second.x),mul(first.y,second.y),
              mul(first.z,second.z),mul(first.w,second.w));
"""),
    "transparency": _evaluate(r"""
    D4 col=load4(0,pos);
    col.w=mul(col.w,U_alpha);
"""),
    "invert": _evaluate(r"""
    D4 source=load4(0,pos);
    D4 col=D4(sub(c(1.0),source.x),sub(c(1.0),source.y),
              sub(c(1.0),source.z),source.w);
"""),
    "to_rgb": _evaluate("D4 col=labToRgb(load4(0,pos));"),
    "to_lab": _evaluate("D4 col=rgbToLab(load4(0,pos));"),
    "dithering": _BAYER
    + _evaluate(r"""
    D4 col=load4(0,pos);
    float threshold=effectsThreshold(pos);
    col.x=add(col.x,c(col.x.v<threshold?0.0:1.0/255.0));
    col.y=add(col.y,c(col.y.v<threshold?0.0:1.0/255.0));
    col.z=add(col.z,c(col.z.v<threshold?0.0:1.0/255.0));
"""),
    "dither_1bit": _BAYER
    + _evaluate(r"""
    D4 source=load4(0,pos);
    float luminance=source.x.v*0.299+source.y.v*0.587+source.z.v*0.114;
    D bit=c(luminance<effectsThreshold(pos)?0.0:1.0);
    D4 col=D4(bit,bit,bit,c(1.0));
"""),
}


def _blur(component, count):
    # Keep the two weighted additions separate and in source order. The sign
    # test uses the loop index (not offset), and factor stays negative once set.
    delta = (
        "ivec2(int(U_offset(i).v),0)"
        if component == "x"
        else "ivec2(0,int(U_offset(i).v))"
    )
    return _evaluate(
        """
    D4 col=mul4(load4(0,pos),U_weight(0));
    int factor=1;
    for(int i=1;i<%d;i++) {
        ivec2 delta=%s;
        col=add4(col,mul4(load4(0,abs(pos-delta)),U_weight(i)));
        if(pos.%s+i>=size.%s-1) factor=-1;
        col=add4(col,mul4(load4(0,pos+factor*delta),U_weight(i)));
    }
"""
        % (count, delta, component, component)
    )


SOURCES.update(
    {
        "blur_{}_{}".format(direction, taps): _blur(component, count)
        for direction, component in (("hor", "x"), ("ver", "y"))
        for taps, count in ((9, 5), (13, 8))
    }
)
