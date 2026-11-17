#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, binding = 0) writeonly uniform image2D destTex;
layout (rgba8, binding = 1) readonly uniform image2D tex0;
layout (rgba8, binding = 2) readonly uniform image2D tex1;
layout (rgba8, binding = 3) readonly uniform image2D mask;
uniform float mode;


void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 destSize = imageSize(destTex);
    if (texelPos.x >= destSize.x || texelPos.y >= destSize.y) {
        return;
    }

    vec4 col0 = imageLoad(tex0, texelPos);
    vec4 col1 = imageLoad(tex1, texelPos);
    float alpha = imageLoad(mask, texelPos).r;

    vec4 mixed = mix(col1, col0, alpha);
    vec4 col = mixed;
    if (mode > 0.5 && mode < 1.5) {
        col = mix(col1, col0 * col1, alpha);
    } else if (mode > 1.5 && mode < 2.5) {
        col = mix(col1, vec4(1.0) - (vec4(1.0) - col0) * (vec4(1.0) - col1), alpha);
    } else if (mode > 2.5) {
        float topAlpha = col0.a * alpha;
        float outAlpha = topAlpha + col1.a * (1.0 - topAlpha);
        vec3 premultiplied = col0.rgb * topAlpha + col1.rgb * col1.a * (1.0 - topAlpha);
        col = vec4(outAlpha > 0.0 ? premultiplied / outAlpha : vec3(0.0), outAlpha);
    }

    imageStore(destTex, texelPos, col);
}
