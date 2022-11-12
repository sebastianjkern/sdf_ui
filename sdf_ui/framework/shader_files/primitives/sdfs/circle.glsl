#version 430
#define SMOOTHSTEP_OFFSET 0.0001
#define PI 3.1415926538
#define AA_DISTANCE 1.5

layout (local_size_x = 16, local_size_y = 16) in;

layout(r32f, location=0) writeonly uniform image2D destTex;

uniform vec2 offset;
uniform float radius;

float sdf_circle(in vec2 p, in float r) {
    return length(p)-r;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    float distance = sdf_circle(texelPos - offset, radius);

    imageStore(destTex, texelPos, vec4(distance, .0, .0, .0));
}
