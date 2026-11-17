#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f, binding=0) writeonly uniform image2D destTex;
layout(rgba32f, binding=1) readonly uniform image2D field0;
uniform float amount;
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,imageSize(destTex)))) return;
    imageStore(destTex,p,imageLoad(field0,p)*amount);
}
