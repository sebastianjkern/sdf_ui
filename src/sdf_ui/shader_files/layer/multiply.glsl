#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, location = 0) writeonly uniform image2D destTex;
layout (rgba8, location = 1) readonly uniform image2D mask1;
layout (rgba8, location = 2) readonly uniform image2D mask2;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    vec4 col0 = imageLoad(mask1, texelPos);
    vec4 col1 = imageLoad(mask2, texelPos);

    imageStore(destTex, texelPos, col0 * col1);
}
