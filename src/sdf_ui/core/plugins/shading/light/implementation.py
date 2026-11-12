"""Light shader's derivative contract and native GPU program."""

from sdf_ui.core.differentiation import ShaderArgument, ShaderDifferentiation
from sdf_ui.core.plugins.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.core.shader_expression import ShaderProgram
from ..glsl import RGBA
from .reference import reference

INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("sdf", 1, read=True),),
    (Uniform("fgColor", "vec4"), Uniform("bgColor", "vec4"), Uniform("lightDir", "vec3"),
     *(Uniform(name, "float") for name in (
         "ambient", "diffuse", "specular", "shininess", "normalStrength", "bevel",
         "shadeBackground", "backgroundBevel", "heightProfile", "heightGamma",
         "height", "inflate", "inner", "outer",
     ))),
)

CONTRACT = ShaderDifferentiation(
    "light", "core/plugins/shading/light/shader.glsl",
    (ShaderArgument("sdf", (), "r32f"),),
    (ShaderArgument("fgColor", (4,), "vec4"), ShaderArgument("bgColor", (4,), "vec4"),
     ShaderArgument("lightDir", (3,), "vec3"),
     *(ShaderArgument(name, (), "float") for name in (
         "ambient", "diffuse", "specular", "shininess", "normalStrength", "bevel",
         "shadeBackground", "backgroundBevel", "heightProfile", "heightGamma",
         "height", "inflate", "inner", "outer",
     ))),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    zero_gradient_parameters=("heightProfile",),
    notes="Sobel stencil, normals/material/light differentiate; heightProfile is discrete.",
    sampling="clamped_sobel", reference=reference,
)
BODY = """
D shadingPositivePower(D base, D exponent) {
    if (base.v <= 0.0) return c(0.0);
    return powD(base, exponent);
}
D shadingHeightCurve(D t) {
    t = shadingPositivePower(clampD(t, c(0.0), c(1.0)),
                             maxD(U_heightGamma, c(0.0001)));
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
PROGRAM = ShaderProgram("light", RGBA + BODY)
