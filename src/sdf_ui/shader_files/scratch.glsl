#version 430
#define SMOOTHSTEP_OFFSET 0.0001
#define PI 3.1415926538
#define AA_DISTANCE 1.5

layout (local_size_x = 16, local_size_y = 16) in;

float smin(float a, float b, float k) {
    float res = exp(-k * a) + exp(-k * b);
    return -log(res) / k;
}

vec3 elem_max(vec3 a, vec3 b) {
    return vec3(
        max(a.r, b.r),
        max(a.g, b.g),
        max(a.b, b.b)
    );
}
