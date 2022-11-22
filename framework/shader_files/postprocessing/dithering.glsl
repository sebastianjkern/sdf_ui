#version 430

#define NOISE_GRANULARITY 0.5/255.0

layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba8, location=0) writeonly uniform image2D destTex;
layout(rgba8, location=1) readonly uniform image2D origTex;

float find_closest(int x, int y, float c0)
{
    int dither[8][8] = {
    { 0, 32, 8, 40, 2, 34, 10, 42 }, /* 8x8 Bayer ordered dithering */
    { 48, 16, 56, 24, 50, 18, 58, 26 }, /* pattern. Each input pixel */
    { 12, 44, 4, 36, 14, 46, 6, 38 }, /* is scaled to the 0..63 range */
    { 60, 28, 52, 20, 62, 30, 54, 22 }, /* before looking in this table */
    { 3, 35, 11, 43, 1, 33, 9, 41 }, /* to determine the action. */
    { 51, 19, 59, 27, 49, 17, 57, 25 },
    { 15, 47, 7, 39, 13, 45, 5, 37 },
    { 63, 31, 55, 23, 61, 29, 53, 21 } };

    float limit = 0.0;
    if (x < 8)
    {
        limit = (dither[x][y]+1)/64.0;
    }

    if (c0 < limit) return 0.0;
    return 1.0;
}

float rnd(vec2 coords) {
    return fract(sin(dot(coords.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    int x = int(texelPos.x % 8);
    int y = int(texelPos.y % 8);

    vec4 col = imageLoad(origTex, texelPos);

    col.r += find_closest(x, y, col.x) / 255;
    col.g += find_closest(x, y, col.y) / 255;
    col.b += find_closest(x, y, col.z) / 255;

    imageStore(destTex, texelPos, col);
}
