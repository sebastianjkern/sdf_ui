#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f, binding=0) writeonly uniform image2D destTex;
uniform vec2 vector;
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,imageSize(destTex)))) return;
    imageStore(destTex,p,vec4(vector,0.0,0.0));
}
