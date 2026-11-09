"""Analytic dual GLSL bodies for the nine standalone shading shaders.

``SOURCES[descriptor.name]`` defines ``evaluate(ivec2 pos, out D o[8])``.
The first four outputs are unquantized RGBA; the remaining slots are zero.
The shared emitter supplies dual arithmetic, uniforms, image loads, and color
conversion. Image indices follow the descriptor's input declaration order.
These shared programs generate both the native forward and GPU VJP shaders.
"""

__all__ = ["SOURCES"]


_RGBA = """
void shadingOutput(D4 color, out D o[8]) {
    o[0] = color.x; o[1] = color.y; o[2] = color.z; o[3] = color.w;
    for (int i = 4; i < 8; ++i) o[i] = c(0.0);
}
D4 shadingMix(D4 a, D4 b, D t) {
    return D4(mixD(a.x, b.x, t), mixD(a.y, b.y, t),
              mixD(a.z, b.z, t), mixD(a.w, b.w, t));
}
"""


_CLEAR_COLOR = """
void evaluate(ivec2 pos, out D o[8]) {
    shadingOutput(rgbToLab(U_color), o);
}
"""


_FILL = """
void evaluate(ivec2 pos, out D o[8]) {
    D distance = sub(ld(0, pos, 0), U_inflate);
    D weight = smoothD(U_first, add(U_second, c(0.0001)), distance);
    shadingOutput(shadingMix(rgbToLab(U_color), rgbToLab(U_background), weight), o);
}
"""


_OUTLINE = """
void evaluate(ivec2 pos, out D o[8]) {
    D distance = absD(sub(ld(0, pos, 0), U_inflate));
    // Native OUTLINE_DISTANCE / 2 is integer division, giving zero.
    D weight = smoothD(c(0.0), c(1.5001), distance);
    shadingOutput(shadingMix(rgbToLab(U_outline), rgbToLab(U_background), weight), o);
}
"""


_FILL_FROM_TEXTURE = """
void evaluate(ivec2 pos, out D o[8]) {
    D distance = sub(ld(1, pos, 0), U_inflate);
    D weight = smoothD(c(0.0), c(1.5001), distance);
    // This shader mixes the supplied texture/color directly, without LAB.
    shadingOutput(shadingMix(load4(0, pos), U_background, weight), o);
}
"""


_ISOLINES = """
void evaluate(ivec2 pos, out D o[8]) {
    D distance = absD(sub(ld(0, pos, 0), U_inflate));
    D4 background = rgbToLab(U_background);
    if (isnan(distance.v) || distance.v > 1e20) {
        shadingOutput(background, o);
        return;
    }
    D period = maxD(U_spacing, c(1e-6));
    D wrapped = modD(add(distance, U_phase), period);
    if (wrapped.v < 0.0) wrapped = add(wrapped, period);
    D band = minD(wrapped, sub(period, wrapped));
    D halfWidth = maxD(mul(U_line_width, c(0.5)), c(0.0));
    D aa = add(c(1.5001), maxD(U_feather, c(0.0)));
    D weight = sub(c(1.0), smoothD(maxD(sub(halfWidth, aa), c(0.0)),
                                  add(halfWidth, aa), band));
    shadingOutput(shadingMix(background, rgbToLab(U_line_color), weight), o);
}
"""


_PARTIAL_DERIVATIVE = """
void evaluate(ivec2 pos, out D o[8]) {
    D center = ld(0, pos, 0);
    shadingOutput(D4(sub(center, ld(0, pos + ivec2(1, 0), 0)),
                     sub(center, ld(0, pos + ivec2(0, 1), 0)),
                     c(0.0), c(1.0)), o);
}
"""


_LIGHT = """
// pow(0, gamma) has a finite zero subgradient, including gamma < 1.
D shadingPositivePower(D base, D exponent) {
    if (base.v <= 0.0) return c(0.0);
    return powD(base, exponent);
}
D shadingHeightCurve(D t) {
    t = shadingPositivePower(clampD(t, c(0.0), c(1.0)),
                             maxD(U_heightGamma, c(0.0001)));
    // The profile selects a formula; its selector has no derivative.
    if (U_heightProfile.v < 0.5) return smoothD(c(0.0), c(1.0), t);
    if (U_heightProfile.v < 1.5)
        return sub(c(0.5), mul(c(0.5), cosD(mul(t, c(3.1415926538)))));
    if (U_heightProfile.v < 2.5) return sqr(t);
    if (U_heightProfile.v < 3.5) return sub(c(1.0), sqr(sub(c(1.0), t)));
    if (U_heightProfile.v < 4.5)
        return sqrtD(maxD(c(0.0), sub(c(1.0), sqr(sub(c(1.0), t)))));
    return sub(c(1.0), sqrtD(maxD(c(0.0), sub(c(1.0), sqr(t)))));
}
D shadingDistance(ivec2 pos) {
    return sub(ld(0, clamp(pos, ivec2(0), size - ivec2(1)), 0), U_inflate);
}
D shadingCoverage(D distance) {
    return sub(c(1.0), smoothD(U_inner, add(U_outer, c(0.0001)), distance));
}
D shadingHeight(ivec2 pos) {
    D distance = shadingDistance(pos);
    if (U_shadeBackground.v < 0.5) return shadingCoverage(distance);
    D bevel = maxD(U_bevel, c(0.0001));
    D bgBevel = maxD(U_backgroundBevel.v > 0.0 ? U_backgroundBevel : bevel,
                    c(0.0001));
    D t = mul(U_shadeBackground, sub(c(1.0),
        smoothD(c(0.0), bgBevel, maxD(distance, c(0.0)))));
    return shadingHeightCurve(t);
}
void evaluate(ivec2 pos, out D o[8]) {
    D coverage = shadingCoverage(shadingDistance(pos));
    D h00 = shadingHeight(pos + ivec2(-1, -1));
    D h10 = shadingHeight(pos + ivec2( 0, -1));
    D h20 = shadingHeight(pos + ivec2( 1, -1));
    D h01 = shadingHeight(pos + ivec2(-1,  0));
    D h21 = shadingHeight(pos + ivec2( 1,  0));
    D h02 = shadingHeight(pos + ivec2(-1,  1));
    D h12 = shadingHeight(pos + ivec2( 0,  1));
    D h22 = shadingHeight(pos + ivec2( 1,  1));
    D gx = mul(sub(add(add(h20, mul(c(2.0), h21)), h22),
                   add(add(h00, mul(c(2.0), h01)), h02)), c(0.125));
    D gy = mul(sub(add(add(h02, mul(c(2.0), h12)), h22),
                   add(add(h00, mul(c(2.0), h10)), h20)), c(0.125));
    D sx = clampD(neg(mul(gx, U_normalStrength)), c(-8.0), c(8.0));
    D sy = clampD(neg(mul(gy, U_normalStrength)), c(-8.0), c(8.0));
    D3 normal = D3(sx, sy, maxD(U_height, c(0.0001)));
    normal = div3(normal, len3(normal));
    D3 light = D3(U_lightDir.x, neg(U_lightDir.y), U_lightDir.z);
    light = div3(light, len3(light));
    D ndotl = dot3(normal, light);
    D lambert = maxD(ndotl, c(0.0));
    D highlight = c(0.0);
    if (lambert.v > 0.0 && U_specular.v > 0.0) {
        D reflectedZ = maxD(sub(mul(mul(c(2.0), ndotl), normal.z), light.z), c(0.0));
        highlight = shadingPositivePower(reflectedZ, maxD(U_shininess, c(0.0001)));
    }
    D shade = add(maxD(U_ambient, c(0.0)), mul(maxD(U_diffuse, c(0.0)), lambert));
    D spec = mul(maxD(U_specular, c(0.0)), highlight);
    D4 base = shadingMix(U_bgColor, U_fgColor, coverage);
    D4 lit = D4(clampD(add(mul(base.x, shade), spec), c(0.0), c(1.0)),
                clampD(add(mul(base.y, shade), spec), c(0.0), c(1.0)),
                clampD(add(mul(base.z, shade), spec), c(0.0), c(1.0)), base.w);
    D h = shadingHeight(pos);
    D slope = mul(len2(D2(gx, gy)), U_normalStrength);
    D heightBand = mul(smoothD(c(0.0), c(0.08), h),
                       smoothD(c(0.0), c(0.08), sub(c(1.0), h)));
    D influence = mul(heightBand, smoothD(c(0.0), c(0.08), slope));
    shadingOutput(shadingMix(base, lit, influence), o);
}
"""


_FILM_GRAIN = """
float shadingNoise(vec2 p) {
    vec2 K1 = vec2(23.14069263277926, 2.665144142690225);
    return fract(cos(dot(p, K1)) * 12345.6789);
}
void evaluate(ivec2 pos, out D o[8]) {
    // Pixel coordinates are fixed and the native time uniform is unused.
    D grain = c(shadingNoise(vec2(pos)));
    shadingOutput(D4(grain, grain, grain, c(1.0)), o);
}
"""


_PERLIN_NOISE = """
// The native classic Perlin functions operate on fixed float coordinates.
vec4 shadingMod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}
vec4 shadingPermute(vec4 x) {
    return shadingMod289(((x * 34.0) + 10.0) * x);
}
vec4 shadingTaylorInvSqrt(vec4 r) {
    return 1.79284291400159 - 0.85373472095314 * r;
}
vec2 shadingFade(vec2 t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}
float shadingCnoise(vec2 P) {
    vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
    vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
    Pi = shadingMod289(Pi);
    vec4 ix = Pi.xzxz;
    vec4 iy = Pi.yyww;
    vec4 fx = Pf.xzxz;
    vec4 fy = Pf.yyww;
    vec4 i = shadingPermute(shadingPermute(ix) + iy);
    vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0;
    vec4 gy = abs(gx) - 0.5;
    vec4 tx = floor(gx + 0.5);
    gx = gx - tx;
    vec2 g00 = vec2(gx.x, gy.x);
    vec2 g10 = vec2(gx.y, gy.y);
    vec2 g01 = vec2(gx.z, gy.z);
    vec2 g11 = vec2(gx.w, gy.w);
    vec4 norm = shadingTaylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01),
                                          dot(g10, g10), dot(g11, g11)));
    g00 *= norm.x;
    g01 *= norm.y;
    g10 *= norm.z;
    g11 *= norm.w;
    float n00 = dot(g00, vec2(fx.x, fy.x));
    float n10 = dot(g10, vec2(fx.y, fy.y));
    float n01 = dot(g01, vec2(fx.z, fy.z));
    float n11 = dot(g11, vec2(fx.w, fy.w));
    vec2 fade_xy = shadingFade(Pf.xy);
    vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
    float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}
void evaluate(ivec2 pos, out D o[8]) {
    D value = c(shadingCnoise(1.5 * (vec2(pos) / 256.0)));
    shadingOutput(D4(value, value, value, c(1.0)), o);
}
"""


SOURCES = {
    name: _RGBA + body
    for name, body in {
        "clear_color": _CLEAR_COLOR,
        "fill": _FILL,
        "fill_from_texture": _FILL_FROM_TEXTURE,
        "film_grain": _FILM_GRAIN,
        "isolines": _ISOLINES,
        "light": _LIGHT,
        "outline": _OUTLINE,
        "partial_derivative": _PARTIAL_DERIVATIVE,
        "perlin_noise": _PERLIN_NOISE,
    }.items()
}
