#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f, binding=0) writeonly uniform image2D destTex;
uniform vec2 center;
uniform float radius;
uniform float strength;
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,imageSize(destTex)))) return;
    vec2 delta=vec2(p)-center;
    float attenuation=exp(-dot(delta,delta)/(2.0*radius*radius));
    imageStore(destTex,p,vec4(delta/radius*strength*attenuation,0.0,0.0));
}
