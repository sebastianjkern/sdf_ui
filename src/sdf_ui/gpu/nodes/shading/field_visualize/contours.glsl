#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba8,binding=0) writeonly uniform image2D destTex;
layout(rgba32f,binding=1) readonly uniform image2D fieldTex;
uniform float spacing;
uniform float line_width;
uniform float scale;
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,imageSize(destTex)))) return;
    float value=length(imageLoad(fieldTex,p).rg)*scale;
    float band=abs(mod(value+spacing*0.5,spacing)-spacing*0.5);
    float line=1.0-smoothstep(max(line_width-0.75,0.0),line_width+0.75,band);
    vec3 color=mix(vec3(0.035,0.055,0.095),vec3(0.27,0.88,0.92),line);
    imageStore(destTex,p,vec4(color,1.0));
}
