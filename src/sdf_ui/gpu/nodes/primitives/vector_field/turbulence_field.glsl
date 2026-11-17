#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f, binding=0) writeonly uniform image2D destTex;
uniform float frequency;
uniform float amplitude;
uniform int octaves;
uniform float lacunarity;
uniform float gain;
uniform int seed;
float hash(ivec2 p,int s) {
    uint h=uint(p.x)*0x8da6b343u ^ uint(p.y)*0xd8163841u ^ uint(s)*0xcb1ab31fu;
    h^=h>>16; h*=0x7feb352du; h^=h>>15; h*=0x846ca68bu; h^=h>>16;
    return float(h&0x00ffffffu)/16777215.0;
}
float valueNoise(vec2 p,int s) {
    ivec2 i=ivec2(floor(p)); vec2 f=fract(p); f=f*f*(3.0-2.0*f);
    float a=hash(i,s), b=hash(i+ivec2(1,0),s), c=hash(i+ivec2(0,1),s), d=hash(i+ivec2(1,1),s);
    return mix(mix(a,b,f.x),mix(c,d,f.x),f.y)*2.0-1.0;
}
float fbm(vec2 p,int seedOffset) {
    float sum=0.0, weight=1.0, total=0.0, f=frequency;
    for(int i=0;i<8;i++) {
        if(i>=octaves) break;
        sum+=valueNoise(p*f,seed+i+int(seedOffset))*weight;
        total+=weight; weight*=gain; f*=lacunarity;
    }
    return sum/max(total,0.0001);
}
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,imageSize(destTex)))) return;
    vec2 xy=vec2(p);
    imageStore(destTex,p,vec4(vec2(fbm(xy,0),fbm(xy+vec2(37.1,-19.7),101))*amplitude,0.0,0.0));
}
