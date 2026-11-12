#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, binding = 0) writeonly uniform image2D destTex;

uniform float time = 1.0;

float noise(vec2 p)
{
    uvec2 pixel = uvec2(p);
    uint value = pixel.x * 0x9E3779B9u + pixel.y * 0x85EBCA6Bu;
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    value ^= value >> 16;
    return float(value & 0x00FFFFFFu) / 16777216.0;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 destSize = imageSize(destTex);
    if (texelPos.x >= destSize.x || texelPos.y >= destSize.y) {
        return;
    }

    float grain = noise(vec2(texelPos));

    imageStore(destTex, texelPos, vec4(vec3(grain), 1.0));
}
