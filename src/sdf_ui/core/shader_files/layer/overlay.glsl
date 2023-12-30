#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, location = 0) writeonly uniform image2D destTex;
layout (rgba8, location = 1) readonly uniform image2D tex0;
layout (rgba8, location = 2) readonly uniform image2D tex1;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    vec4 col0 = imageLoad(tex0, texelPos);
    vec4 col1 = imageLoad(tex1, texelPos);

    float ac = col0.a + (1 - col0.a) * col1.a;

    vec3 col = 1 / ac * (col0.rgb * col0.a + col1.rgb * (1.0 - col0.a) * col1.a);
    vec4 o_col = vec4(col, ac);

    imageStore(destTex, texelPos, o_col);
}
