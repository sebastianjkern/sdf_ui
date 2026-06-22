#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (r32f, binding = 0) writeonly uniform image2D destTex;
layout (r32f, binding = 1) readonly uniform image2D sdf;

uniform float transform_kind;
uniform vec2 offset;
uniform vec2 center;
uniform float factor;
uniform float angle;
uniform vec2 skew;

float sample_sdf(vec2 p) {
    ivec2 size = imageSize(sdf);
    vec2 max_pos = vec2(size) - vec2(1.0);
    vec2 pos = clamp(p, vec2(0.0), max_pos);

    ivec2 p0 = ivec2(floor(pos));
    ivec2 p1 = min(p0 + ivec2(1, 0), size - ivec2(1, 1));
    ivec2 p2 = min(p0 + ivec2(0, 1), size - ivec2(1, 1));
    ivec2 p3 = min(p0 + ivec2(1, 1), size - ivec2(1, 1));

    float tx = pos.x - float(p0.x);
    float ty = pos.y - float(p0.y);

    float a = imageLoad(sdf, p0).r;
    float b = imageLoad(sdf, p1).r;
    float c = imageLoad(sdf, p2).r;
    float d = imageLoad(sdf, p3).r;

    return mix(mix(a, b, tx), mix(c, d, tx), ty);
}

vec2 inverse_transform(vec2 p) {
    if (transform_kind < 0.5) {
        return p - offset;
    }

    vec2 local = p - center;

    if (transform_kind < 1.5) {
        return center + local / factor;
    }

    if (transform_kind < 2.5) {
        float c = cos(angle);
        float s = sin(angle);
        mat2 inv_rotation = mat2(c, -s, s, c);
        return center + inv_rotation * local;
    }

    float det = 1.0 - skew.x * skew.y;
    mat2 inv_skew = mat2(1.0, -skew.y, -skew.x, 1.0) / det;
    return center + inv_skew * local;
}

float distance_scale() {
    if (transform_kind < 1.5) {
        return abs(factor);
    }

    if (transform_kind > 2.5) {
        float det = 1.0 - skew.x * skew.y;
        mat2 inv_skew = mat2(1.0, -skew.y, -skew.x, 1.0) / det;
        float stretch_x = length(inv_skew[0]);
        float stretch_y = length(inv_skew[1]);
        return max(0.5 * (stretch_x + stretch_y), 1e-6);
    }

    return 1.0;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 destSize = imageSize(destTex);
    if (texelPos.x >= destSize.x || texelPos.y >= destSize.y) {
        return;
    }

    vec2 source_pos = inverse_transform(vec2(texelPos));
    float distance = sample_sdf(source_pos) * distance_scale();

    imageStore(destTex, texelPos, vec4(distance, .0, .0, .0));
}
