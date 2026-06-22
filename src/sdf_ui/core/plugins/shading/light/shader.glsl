#version 430
#define SMOOTHSTEP_OFFSET 0.0001

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, binding = 0) writeonly uniform image2D destTex;
layout (r32f, binding = 1) readonly uniform image2D sdf;

uniform vec4 fgColor;
uniform vec4 bgColor;
uniform vec3 lightDir;
uniform float ambient;
uniform float diffuse;
uniform float specular;
uniform float shininess;
uniform float normalStrength;
uniform float bevel;
uniform float height;
uniform float inflate;
uniform float inner;
uniform float outer;

float sampleSdf(ivec2 p, ivec2 size) {
    ivec2 clamped = clamp(p, ivec2(0), size - ivec2(1));
    return imageLoad(sdf, clamped).r - inflate;
}

float bevelHeight(ivec2 p, ivec2 size, float bevelWidth) {
    float distance = sampleSdf(p, size);
    return 1.0 - smoothstep(-bevelWidth, 0.0, distance);
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 destSize = imageSize(destTex);
    if (texelPos.x >= destSize.x || texelPos.y >= destSize.y) {
        return;
    }

    float distance = sampleSdf(texelPos, destSize);
    float outside = smoothstep(inner, outer + SMOOTHSTEP_OFFSET, distance);
    float coverage = 1.0 - outside;

    float bevelWidth = max(bevel, 0.0001);
    float h00 = bevelHeight(texelPos + ivec2(-1, -1), destSize, bevelWidth);
    float h10 = bevelHeight(texelPos + ivec2( 0, -1), destSize, bevelWidth);
    float h20 = bevelHeight(texelPos + ivec2( 1, -1), destSize, bevelWidth);
    float h01 = bevelHeight(texelPos + ivec2(-1,  0), destSize, bevelWidth);
    float h21 = bevelHeight(texelPos + ivec2( 1,  0), destSize, bevelWidth);
    float h02 = bevelHeight(texelPos + ivec2(-1,  1), destSize, bevelWidth);
    float h12 = bevelHeight(texelPos + ivec2( 0,  1), destSize, bevelWidth);
    float h22 = bevelHeight(texelPos + ivec2( 1,  1), destSize, bevelWidth);

    float dhdx = ((h20 + 2.0 * h21 + h22) - (h00 + 2.0 * h01 + h02)) * 0.125;
    float dhdy = ((h02 + 2.0 * h12 + h22) - (h00 + 2.0 * h10 + h20)) * 0.125;
    vec2 slope = clamp(vec2(-dhdx, -dhdy) * normalStrength, vec2(-8.0), vec2(8.0));
    vec3 normal = normalize(vec3(slope, max(height, 0.0001)));
    vec3 light = normalize(vec3(lightDir.x, -lightDir.y, lightDir.z));
    vec3 view = vec3(0.0, 0.0, 1.0);

    float lambert = max(dot(normal, light), 0.0);
    float highlight = 0.0;
    if (lambert > 0.0 && specular > 0.0) {
        vec3 reflected = reflect(-light, normal);
        highlight = pow(max(dot(reflected, view), 0.0), max(shininess, 0.0001));
    }

    float shade = max(ambient, 0.0) + max(diffuse, 0.0) * lambert;
    vec3 lit = fgColor.rgb * shade + vec3(max(specular, 0.0) * highlight);
    vec4 fg = vec4(clamp(lit, 0.0, 1.0), fgColor.a);
    vec4 col = mix(bgColor, fg, coverage);

    imageStore(destTex, texelPos, col);
}
