#version 430
#define SMOOTHSTEP_OFFSET 0.0001

layout (local_size_x = 16, local_size_y = 16) in;

layout(r32f, location=0) writeonly uniform image2D destTex;

uniform vec2 size;
uniform vec2 offset;
uniform vec4 corner_radius;

float sdf_roundbox(in vec2 uv, in vec2 size, in vec4 corner_radius)
{
    corner_radius.xy = mix(corner_radius.xy, corner_radius.zw, smoothstep(0, SMOOTHSTEP_OFFSET, uv.x));
    corner_radius.x = mix(corner_radius.x, corner_radius.y, smoothstep(0, SMOOTHSTEP_OFFSET, uv.y));
    vec2 q = abs(uv)-size+corner_radius.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - corner_radius.x;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    float distance = sdf_roundbox(texelPos - offset, size, corner_radius);

    imageStore(destTex, texelPos, vec4(distance, .0, .0, .0));
}
