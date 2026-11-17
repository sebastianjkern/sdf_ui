#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba8,binding=0) writeonly uniform image2D destTex;
layout(rgba32f,binding=1) readonly uniform image2D fieldTex;
uniform float scale;
uniform float magnitude;
vec3 hsv(float h,float s,float v) {
    vec3 p=abs(fract(vec3(h)+vec3(0.0,2.0/3.0,1.0/3.0))*6.0-3.0);
    return v*mix(vec3(1.0),clamp(p-1.0,0.0,1.0),s);
}
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,imageSize(destTex)))) return;
    vec2 v=imageLoad(fieldTex,p).rg*scale;
    float hue=atan(v.y,v.x)/6.28318530718+0.5;
    float brightness=1.0-exp(-length(v)*magnitude);
    imageStore(destTex,p,vec4(hsv(hue,0.78,brightness),1.0));
}
