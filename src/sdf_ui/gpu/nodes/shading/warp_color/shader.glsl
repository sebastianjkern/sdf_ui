#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba8,binding=0) writeonly uniform image2D destTex;
layout(rgba8,binding=1) readonly uniform image2D sourceTex;
layout(rgba32f,binding=2) readonly uniform image2D fieldTex;
uniform float strength;
uniform int steps;
vec2 fieldAt(vec2 p) {
    ivec2 size=imageSize(fieldTex); p=clamp(p,vec2(0),vec2(size-1));
    ivec2 a=ivec2(floor(p)), b=min(a+1,size-1); vec2 f=fract(p);
    vec2 v00=imageLoad(fieldTex,a).rg, v10=imageLoad(fieldTex,ivec2(b.x,a.y)).rg;
    vec2 v01=imageLoad(fieldTex,ivec2(a.x,b.y)).rg, v11=imageLoad(fieldTex,b).rg;
    return mix(mix(v00,v10,f.x),mix(v01,v11,f.x),f.y);
}
vec4 sourceAt(vec2 p) {
    ivec2 size=imageSize(sourceTex);
    if(any(lessThan(p,vec2(0)))||any(greaterThan(p,vec2(size-1)))) return vec4(0.0);
    p=clamp(p,vec2(0),vec2(size-1)); ivec2 a=ivec2(floor(p)), b=min(a+1,size-1); vec2 f=fract(p);
    vec4 v00=imageLoad(sourceTex,a), v10=imageLoad(sourceTex,ivec2(b.x,a.y));
    vec4 v01=imageLoad(sourceTex,ivec2(a.x,b.y)), v11=imageLoad(sourceTex,b);
    return mix(mix(v00,v10,f.x),mix(v01,v11,f.x),f.y);
}
void main() {
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(pixel,imageSize(destTex)))) return;
    vec2 position=vec2(pixel), stepSize=strength/float(steps);
    for(int i=0;i<steps;i++) position-=fieldAt(position)*stepSize;
    imageStore(destTex,pixel,sourceAt(position));
}
