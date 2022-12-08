#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba8, location=0) writeonly uniform image2D destTex;

uniform float time = 1.0;

float noise( vec2 p )
{
    vec2 K1 = vec2(
    23.14069263277926, // e^pi (Gelfond's constant)
    2.665144142690225 // 2^sqrt(2) (Gelfond–Schneider constant)
    );
    return fract( cos( dot(p,K1) ) * 12345.6789 );
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    float grain = noise(vec2(texelPos));

    imageStore(destTex, texelPos, vec4(vec3(grain), 1.0));
}
