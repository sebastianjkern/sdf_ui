#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, location = 0) writeonly uniform image2D destTex;
layout (rgba8, location = 1) readonly uniform image2D origTex;


void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    vec4 col = imageLoad(origTex, texelPos);
    vec4 out_col = vec4(vec3(1) - col.rgb, col.a);

    imageStore(destTex, texelPos, out_col);
}