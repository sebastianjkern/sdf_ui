#version 430
#define SMOOTHSTEP_OFFSET 0.0001
#define PI 3.1415926538
#define AA_DISTANCE 1.5

layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba8, location=0) writeonly uniform image2D destTex;
layout(r32f, location=1) readonly uniform image2D sdf;

uniform float inflate;
uniform vec4 color;
uniform vec4 background;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    float distance = imageLoad(sdf, texelPos).r - inflate;

    float fill = smoothstep(0-AA_DISTANCE-SMOOTHSTEP_OFFSET, 0, distance);

    vec4 col = mix(color, background, fill);

    imageStore(destTex, texelPos, col);
}
