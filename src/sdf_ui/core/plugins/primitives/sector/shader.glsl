#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (r32f, binding = 0) writeonly uniform image2D destTex;

uniform vec2 center;
uniform float radius;
uniform float start_angle;
uniform float end_angle;

const float TAU = 6.28318530718;

float normalize_angle(float angle) {
    angle = mod(angle, TAU);
    return angle < 0.0 ? angle + TAU : angle;
}

bool angle_in_range(float angle, float start, float end) {
    if (start <= end) {
        return angle >= start && angle <= end;
    }
    return angle >= start || angle <= end;
}

float distance_to_segment(vec2 p, vec2 a, vec2 b) {
    vec2 ab = b - a;
    float denom = max(dot(ab, ab), 0.000001);
    float h = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
    return length(p - (a + h * ab));
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 destSize = imageSize(destTex);
    if (texelPos.x >= destSize.x || texelPos.y >= destSize.y) {
        return;
    }

    vec2 p = vec2(texelPos) - center;
    float angle = normalize_angle(atan(p.y, p.x));
    float start = normalize_angle(start_angle);
    float end = normalize_angle(end_angle);
    vec2 start_point = vec2(cos(start), sin(start)) * radius;
    vec2 end_point = vec2(cos(end), sin(end)) * radius;

    float circle_distance = abs(length(p) - radius);
    float start_distance = distance_to_segment(p, vec2(0.0), start_point);
    float end_distance = distance_to_segment(p, vec2(0.0), end_point);
    float distance = min(circle_distance, min(start_distance, end_distance));

    if (angle_in_range(angle, start, end) && length(p) <= radius) {
        distance = -distance;
    }

    imageStore(destTex, texelPos, vec4(distance, 0.0, 0.0, 0.0));
}

