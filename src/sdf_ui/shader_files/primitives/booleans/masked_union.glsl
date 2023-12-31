#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (r32f, location = 0) writeonly uniform image2D destTex;
layout (rgba8, location = 1) writeonly uniform image2D maskTex;
layout (r32f, location = 2) readonly uniform image2D sdf0;
layout (r32f, location = 3) readonly uniform image2D sdf1;

#define K 0.1

float sstep(float x) {
    return x/(K + abs(x));
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    float d0 = imageLoad(sdf0, texelPos).r;
    float d1 = imageLoad(sdf1, texelPos).r;

    float distance = min(d0, d1);

    // TODO: The threshold needs work, either causes weird lines or masks don't have antialiasing
    float mask = smoothstep(0, 1, d1 - distance);

    imageStore(destTex, texelPos, vec4(distance, .0, .0, .0));
    imageStore(maskTex, texelPos, vec4(vec3(mask), 1.0));
}