#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (r32f, location = 0) writeonly uniform image2D destTex;

uniform vec2 a;
uniform vec2 b;

float sdf_linesegment(vec2 uv, vec2 a, vec2 b)
{
    vec2 ba = b - a;
    vec2 pa = uv - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    float position = sign((b.x - a.x) * (uv.y - a.y) - (b.y - a.y) * (uv.x - a.x));
    return length(pa - h * ba) * position;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    float distance = sdf_linesegment(texelPos, a, b);

    imageStore(destTex, texelPos, vec4(distance, .0, .0, .0));
}
