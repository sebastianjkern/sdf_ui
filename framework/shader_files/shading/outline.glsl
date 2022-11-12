#version 430
#define SMOOTHSTEP_OFFSET 0.0001
#define PI 3.1415926538
#define AA_DISTANCE 1.5

#define OUTLINE_DISTANCE -1

layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba8, location=0) writeonly uniform image2D destTex;
layout(r32f, location=1) readonly uniform image2D sdf;

uniform vec4 background;
uniform vec4 outline;
uniform float inflate;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    float distance = abs(imageLoad(sdf, texelPos).r-inflate);

    float smooth_d = smoothstep(-OUTLINE_DISTANCE/2, OUTLINE_DISTANCE/2+AA_DISTANCE+SMOOTHSTEP_OFFSET, distance);

    vec4 col = mix(outline, background, smooth_d);

    imageStore(destTex, texelPos, col);
}
