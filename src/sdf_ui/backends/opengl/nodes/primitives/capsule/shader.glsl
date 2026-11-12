#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (r32f, binding = 0) writeonly uniform image2D destTex;

uniform vec2 a;
uniform vec2 b;
uniform float radius;

float sdf_capsule(vec2 p, vec2 a, vec2 b, float r)
{
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 0.000001), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 destSize = imageSize(destTex);
    if (texelPos.x >= destSize.x || texelPos.y >= destSize.y) {
        return;
    }

    float distance = sdf_capsule(vec2(texelPos), a, b, radius);

    imageStore(destTex, texelPos, vec4(distance, 0.0, 0.0, 0.0));
}

