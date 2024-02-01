#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout (r32f, location = 0) writeonly uniform image2D destTex;
layout (r32f, location = 1) readonly uniform image2D sdf0;

uniform float repeat;

float sdf(vec2 p) {
    return imageLoad(sdf0, ivec2(p)).r;
}

// correct way to repeat space every s units
float repeated(vec2 p, float s)
{
    vec2 id = round(p/s);
    vec2  o = sign(p-s*id); // neighbor offset direction
    
    float d = 1e20;
    for( int j=0; j<2; j++ )
    for( int i=0; i<2; i++ )
    {
        vec2 rid = id + vec2(i,j)*o;
        vec2 r = p - s*rid;
        d = min( d, sdf(r));
    }
    return d;
}


void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    float newDistance = repeated(texelPos, repeat);

    imageStore(destTex, texelPos, vec4(newDistance, .0, .0, .0));
}