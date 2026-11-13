"""Fixed-coordinate Perlin shader layout and reference implementation."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)
from sdf_ui.gpu.nodes.shader_expression import ShaderProgram

from ..glsl import RGBA


def reference(context, inputs, params):
    import numpy as np

    from sdf_ui.learning.autodiff import Tensor

    x = context.x.data.astype(np.float32) / np.float32(256.0) * np.float32(1.5)
    y = context.y.data.astype(np.float32) / np.float32(256.0) * np.float32(1.5)
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy

    def mod289(v):
        return v - np.floor(v * np.float32(1.0 / 289.0)) * np.float32(289.0)

    def permute(v):
        return mod289((v * np.float32(34.0) + np.float32(10.0)) * v)

    def corner(dx, dy):
        index = permute(permute(mod289(ix + dx)) + mod289(iy + dy))
        gx = index * np.float32(1.0 / 41.0)
        gx = (gx - np.floor(gx)) * np.float32(2.0) - np.float32(1.0)
        gy = np.abs(gx) - np.float32(0.5)
        gx = gx - np.floor(gx + np.float32(0.5))
        norm = np.float32(1.79284291400159) - np.float32(0.85373472095314) * (
            gx * gx + gy * gy
        )
        return gx * norm * (fx - dx) + gy * norm * (fy - dy)

    def fade(t):
        return t * t * t * (t * (t * np.float32(6.0) - np.float32(15.0)) + np.float32(10.0))

    u, v = fade(fx), fade(fy)
    lower = corner(0, 0) * (1 - u) + corner(1, 0) * u
    upper = corner(0, 1) * (1 - u) + corner(1, 1) * u
    value = np.float32(2.3) * (lower * (1 - v) + upper * v)
    return Tensor(np.stack((value, value, value, np.ones_like(value)), axis=-1))


INTERFACE = ShaderInterface(ImageBinding("destTex", 0, write=True))
CONTRACT = ShaderDifferentiation(
    "perlin_noise", "gpu/nodes/shading/perlin_noise/shader.glsl", (), (),
    (ShaderArgument("destTex", (4,), "rgba8"),), behavior="constant",
    notes="Fixed pixel coordinates/noise constants; no trainable shader inputs.", reference=reference,
)
BODY = """
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
PROGRAM = ShaderProgram("perlin_noise", RGBA + BODY)
