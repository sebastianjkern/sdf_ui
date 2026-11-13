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
uniform float shadeBackground;
uniform float backgroundBevel;
uniform float heightProfile;
uniform float heightGamma;
uniform float height;
uniform float inflate;
uniform float inner;
uniform float outer;

float sampleSdf(ivec2 p, ivec2 size) {
    ivec2 clamped = clamp(p, ivec2(0), size - ivec2(1));
    return imageLoad(sdf, clamped).r - inflate;
}

float heightCurve(float t) {
    t = clamp(t, 0.0, 1.0);
    float gamma = max(heightGamma, 0.0001);
    t = pow(t, gamma);

    if (heightProfile < 0.5) {
        return smoothstep(0.0, 1.0, t);
    }
    if (heightProfile < 1.5) {
        return 0.5 - 0.5 * cos(t * 3.1415926538);
    }
    if (heightProfile < 2.5) {
        return t * t;
    }
    if (heightProfile < 3.5) {
        return 1.0 - (1.0 - t) * (1.0 - t);
    }
    if (heightProfile < 4.5) {
        return sqrt(max(0.0, 1.0 - (1.0 - t) * (1.0 - t)));
    }
    return 1.0 - sqrt(max(0.0, 1.0 - t * t));
}

float heightMap(ivec2 p, ivec2 size, float bevelWidth) {
    float distance = sampleSdf(p, size);
    if (shadeBackground < 0.5) {
        return 1.0 - smoothstep(inner, outer + SMOOTHSTEP_OFFSET, distance);
    }

    float bgBevelWidth = max(backgroundBevel > 0.0 ? backgroundBevel : bevelWidth, 0.0001);
    float t = shadeBackground * (1.0 - smoothstep(0.0, bgBevelWidth, max(distance, 0.0)));
    return heightCurve(t);
}

vec2 sobelHeight(ivec2 texelPos, ivec2 size, float bevelWidth) {
    float h00 = heightMap(texelPos + ivec2(-1, -1), size, bevelWidth);
    float h10 = heightMap(texelPos + ivec2( 0, -1), size, bevelWidth);
    float h20 = heightMap(texelPos + ivec2( 1, -1), size, bevelWidth);
    float h01 = heightMap(texelPos + ivec2(-1,  0), size, bevelWidth);
    float h21 = heightMap(texelPos + ivec2( 1,  0), size, bevelWidth);
    float h02 = heightMap(texelPos + ivec2(-1,  1), size, bevelWidth);
    float h12 = heightMap(texelPos + ivec2( 0,  1), size, bevelWidth);
    float h22 = heightMap(texelPos + ivec2( 1,  1), size, bevelWidth);

    float dhdx = ((h20 + 2.0 * h21 + h22) - (h00 + 2.0 * h01 + h02)) * 0.125;
    float dhdy = ((h02 + 2.0 * h12 + h22) - (h00 + 2.0 * h10 + h20)) * 0.125;
    return vec2(dhdx, dhdy);
}

vec4 shadeColor(vec4 baseColor, vec2 gradient, vec3 light, vec3 view) {
    vec2 slope = clamp(-gradient * normalStrength, vec2(-8.0), vec2(8.0));
    vec3 normal = normalize(vec3(slope, max(height, 0.0001)));

    float lambert = max(dot(normal, light), 0.0);
    float highlight = 0.0;
    if (lambert > 0.0 && specular > 0.0) {
        vec3 reflected = reflect(-light, normal);
        highlight = pow(max(dot(reflected, view), 0.0), max(shininess, 0.0001));
    }

    float shade = max(ambient, 0.0) + max(diffuse, 0.0) * lambert;
    vec3 lit = baseColor.rgb * shade + vec3(max(specular, 0.0) * highlight);
    return vec4(clamp(lit, 0.0, 1.0), baseColor.a);
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 destSize = imageSize(destTex);
    if (texelPos.x >= destSize.x || texelPos.y >= destSize.y) {
        return;
    }

    float distance = sampleSdf(texelPos, destSize);
    float aaCoverage = 1.0 - smoothstep(inner, outer + SMOOTHSTEP_OFFSET, distance);
    float bevelWidth = max(bevel, 0.0001);
    vec2 gradient = sobelHeight(texelPos, destSize, bevelWidth);
    float h = heightMap(texelPos, destSize, bevelWidth);
    vec3 light = normalize(vec3(lightDir.x, -lightDir.y, lightDir.z));
    vec3 view = vec3(0.0, 0.0, 1.0);

    vec4 base = mix(bgColor, fgColor, aaCoverage);
    float slope = length(gradient) * normalStrength;
    float heightBand = smoothstep(0.0, 0.08, h) * smoothstep(0.0, 0.08, 1.0 - h);
    float shadeInfluence = heightBand * smoothstep(0.0, 0.08, slope);
    vec4 lit = shadeColor(base, gradient, light, view);
    vec4 col = mix(base, lit, shadeInfluence);

    imageStore(destTex, texelPos, col);
}
