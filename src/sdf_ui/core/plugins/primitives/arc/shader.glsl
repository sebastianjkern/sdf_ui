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

vec2 point_from_angle(float angle) {
    return vec2(cos(angle), sin(angle));
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

    vec2 start_point = point_from_angle(start) * radius;
    vec2 end_point = point_from_angle(end) * radius;

    float distance;
    if (angle_in_range(angle, start, end)) {
        distance = abs(length(p) - radius);
    } else {
        distance = min(length(p - start_point), length(p - end_point));
    }

    imageStore(destTex, texelPos, vec4(distance, 0.0, 0.0, 0.0));
}

