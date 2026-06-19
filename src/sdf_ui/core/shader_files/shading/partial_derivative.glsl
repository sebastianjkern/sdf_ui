#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, location = 0) writeonly uniform image2D destTex;
layout (r32f, location = 1) readonly uniform image2D sdf;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    float x_pd = imageLoad(sdf, texelPos).r - imageLoad(sdf, texelPos+ivec2(1, 0)).r;
    float y_pd = imageLoad(sdf, texelPos).r - imageLoad(sdf, texelPos+ivec2(0, 1)).r;

    imageStore(destTex, texelPos, vec4(x_pd, y_pd, 0, 1));
}