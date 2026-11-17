#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (r32f, binding = 0) writeonly uniform image2D destTex;
layout (rgba8, binding = 1) writeonly uniform image2D maskTex;
layout (r32f, binding = 2) readonly uniform image2D sdf0;
layout (r32f, binding = 3) readonly uniform image2D sdf1;
uniform float edgeWidth;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 destSize = imageSize(destTex);
    if (texelPos.x >= destSize.x || texelPos.y >= destSize.y) {
        return;
    }

    float d0 = imageLoad(sdf0, texelPos).r;
    float d1 = imageLoad(sdf1, texelPos).r;

    float distance = min(d0, d1);

    float width = max(abs(edgeWidth), 0.0001);
    float mask = smoothstep(-width, width, d1 - d0);

    imageStore(destTex, texelPos, vec4(distance, .0, .0, .0));
    imageStore(maskTex, texelPos, vec4(vec3(mask), 1.0));
}
