#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba8, location=0) writeonly uniform image2D destTex;
layout(rgba8, location=1) readonly uniform image2D tex0;
layout(rgba8, location=2) readonly uniform image2D tex1;
layout(rgba8, location=3) readonly uniform image2D mask;


void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    vec4 col0 = imageLoad(tex0, texelPos);
    vec4 col1 = imageLoad(tex1, texelPos);
    float alpha = imageLoad(mask, texelPos).r;

    vec4 col = col0 * alpha + col1 * (1.0 - alpha);

    imageStore(destTex, texelPos, col);
}
