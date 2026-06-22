#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (r32f, binding = 0) writeonly uniform image2D destTex;

uniform vec2 center;
uniform vec2 radii;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 destSize = imageSize(destTex);
    if (texelPos.x >= destSize.x || texelPos.y >= destSize.y) {
        return;
    }

    vec2 p = vec2(texelPos) - center;
    vec2 r = max(abs(radii), vec2(0.000001));
    vec2 q = p / r;
    float f = dot(q, q) - 1.0;
    vec2 grad = 2.0 * p / (r * r);
    float distance = f / max(length(grad), 0.000001);

    imageStore(destTex, texelPos, vec4(distance, 0.0, 0.0, 0.0));
}

