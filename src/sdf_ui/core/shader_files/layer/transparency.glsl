#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, location = 0) writeonly uniform image2D destTex;
layout (rgba8, location = 1) readonly uniform image2D tex0;

uniform float alpha;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    vec4 col0 = imageLoad(tex0, texelPos);

    imageStore(destTex, texelPos, vec4(col0.rgb, col0.a * alpha));
}
