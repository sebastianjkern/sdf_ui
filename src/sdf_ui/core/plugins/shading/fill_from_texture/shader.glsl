#version 430
#define SMOOTHSTEP_OFFSET 0.0001
#define PI 3.1415926538
#define AA_DISTANCE 1.5

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, binding = 0) writeonly uniform image2D destTex;
layout (rgba8, binding = 1) readonly uniform image2D origTex;
layout (r32f, binding = 2) readonly uniform image2D sdf;

uniform float inflate;

uniform vec4 background;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 destSize = imageSize(destTex);
    if (texelPos.x >= destSize.x || texelPos.y >= destSize.y) {
        return;
    }

    float distance = imageLoad(sdf, texelPos).r - inflate;
    float fill = smoothstep(0, 0 + SMOOTHSTEP_OFFSET + AA_DISTANCE, distance);

    vec4 tex_col = imageLoad(origTex, texelPos);

    vec4 col = mix(tex_col, background, fill);

    imageStore(destTex, texelPos, col);
}
