#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (r32f, binding = 0) writeonly uniform image2D destTex;

uniform vec2 point;
uniform vec2 normal;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 destSize = imageSize(destTex);
    if (texelPos.x >= destSize.x || texelPos.y >= destSize.y) {
        return;
    }

    vec2 n = normal / max(length(normal), 0.000001);
    float distance = dot(vec2(texelPos) - point, n);

    imageStore(destTex, texelPos, vec4(distance, 0.0, 0.0, 0.0));
}

