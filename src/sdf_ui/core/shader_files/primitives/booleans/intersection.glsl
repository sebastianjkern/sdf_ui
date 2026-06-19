#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (r32f, location = 0) writeonly uniform image2D destTex;
layout (r32f, location = 1) readonly uniform image2D sdf0;
layout (r32f, location = 2) readonly uniform image2D sdf1;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    float d0 = imageLoad(sdf0, texelPos).r;
    float d1 = imageLoad(sdf1, texelPos).r;

    float distance = max(d0, d1);

    imageStore(destTex, texelPos, vec4(distance, .0, .0, .0));
}