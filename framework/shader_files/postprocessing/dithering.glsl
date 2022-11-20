#version 430

#define NOISE_GRANULARITY 0.5/255.0

layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba8, location=0) writeonly uniform image2D destTex;
layout(rgba8, location=1) readonly uniform image2D origTex;


float rnd(vec2 coords) {
    return fract(sin(dot(coords.xy, vec2(12.9898,78.233))) * 43758.5453);
}

/*
float rnd(vec2 coords) {
    return noise1(coords.x + coords.y);
}
*/

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    vec4 col = imageLoad(origTex, texelPos);

    col.r += mix(-NOISE_GRANULARITY, NOISE_GRANULARITY, rnd(texelPos / imageSize(origTex)));
    col.g += mix(-NOISE_GRANULARITY, NOISE_GRANULARITY, rnd(texelPos / imageSize(origTex)));
    col.b += mix(-NOISE_GRANULARITY, NOISE_GRANULARITY, rnd(texelPos / imageSize(origTex)));

    imageStore(destTex, texelPos, col);
}
