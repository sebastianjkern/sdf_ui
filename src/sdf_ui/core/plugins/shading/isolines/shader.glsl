#version 430

#define AA_DISTANCE 1.5

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, binding = 0) writeonly uniform image2D destTex;
layout (r32f, binding = 1) readonly uniform image2D sdf;

uniform vec4 background;
uniform vec4 line_color;
uniform float inflate;
uniform float spacing;
uniform float line_width;
uniform float feather;
uniform float phase;

vec3 rgb2xyz(vec3 c)
{
    vec3 tmp;
    tmp.x = (c.r > 0.04045) ? pow((c.r + 0.055) / 1.055, 2.4) : c.r / 12.92;
    tmp.y = (c.g > 0.04045) ? pow((c.g + 0.055) / 1.055, 2.4) : c.g / 12.92,
    tmp.z = (c.b > 0.04045) ? pow((c.b + 0.055) / 1.055, 2.4) : c.b / 12.92;
    const mat3x3 mat = mat3x3(
        0.4124, 0.3576, 0.1805,
        0.2126, 0.7152, 0.0722,
        0.0193, 0.1192, 0.9505
    );
    return 100.0 * tmp * mat;
}

vec3 xyz2lab(vec3 c)
{
    vec3 n = c / vec3(95.047, 100, 108.883);
    vec3 v;
    v.x = (n.x > 0.008856) ? pow(n.x, 1.0 / 3.0) : (7.787 * n.x) + (16.0 / 116.0);
    v.y = (n.y > 0.008856) ? pow(n.y, 1.0 / 3.0) : (7.787 * n.y) + (16.0 / 116.0);
    v.z = (n.z > 0.008856) ? pow(n.z, 1.0 / 3.0) : (7.787 * n.z) + (16.0 / 116.0);
    return vec3((116.0 * v.y) - 16.0, 500.0 * (v.x - v.y), 200.0 * (v.y - v.z));
}

vec3 rgb2lab(vec3 c)
{
    vec3 lab = xyz2lab(rgb2xyz(c));
    return vec3(lab.x / 100.0, 0.5 + 0.5 * (lab.y / 127.0), 0.5 + 0.5 * (lab.z / 127.0));
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 destSize = imageSize(destTex);
    if (texelPos.x >= destSize.x || texelPos.y >= destSize.y) {
        return;
    }

    float distance = abs(imageLoad(sdf, texelPos).r - inflate);
    float period = max(spacing, 1e-6);
    float shifted = distance + phase;
    float wrapped = mod(shifted, period);
    if (wrapped < 0.0) {
        wrapped += period;
    }
    float band = min(wrapped, period - wrapped);
    float line_half_width = max(line_width * 0.5, 0.0);
    float line = 1.0 - smoothstep(
        line_half_width,
        line_half_width + feather + AA_DISTANCE,
        band
    );

    vec4 fg_lab = vec4(rgb2lab(line_color.rgb), line_color.a);
    vec4 bg_lab = vec4(rgb2lab(background.rgb), background.a);
    vec4 col = mix(bg_lab, fg_lab, line);

    imageStore(destTex, texelPos, col);
}

