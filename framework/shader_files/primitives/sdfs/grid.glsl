#version 430

layout (local_size_x = 16, local_size_y = 16) in;

layout(r32f, location=0) writeonly uniform image2D destTex;

uniform vec2 grid_size;
uniform vec2 offset;

float sdf_grid(vec2 uv) {
    return min(abs(mod(uv.x - offset.x, grid_size.x) - grid_size.x/2), abs(mod(uv.y - offset.y, grid_size.y) - grid_size.y/2));
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    float distance = sdf_grid(texelPos);

    imageStore(destTex, texelPos, vec4(distance, .0, .0, .0));
}
