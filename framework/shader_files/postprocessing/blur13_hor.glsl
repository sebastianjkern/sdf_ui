#version 430
#define PI 3.1415926538
#define QUALITY 3.0
#define DIRECTIONS 16.0

#define SIZE 8

layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba8, location=0) writeonly uniform image2D destTex;
layout(rgba8, location=1) readonly uniform image2D origTex;

uniform float offset[SIZE] = float[](0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
uniform float weight[SIZE] = float[](0.2094726562, 0.1832885742, 0.1221923828, 0.0610961914, 0.0222167969, 0.0055541992, 0.0008544922, 0.0000610352);

ivec2 abs(ivec2 x) {
    return ivec2(abs(x.x), abs(x.y));
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(origTex);

    vec4 col = imageLoad(origTex, texelPos) * weight[0];

    int factor = 1;

    for (int i=1; i < SIZE; i++) {
        col += imageLoad(origTex, abs(texelPos - ivec2(offset[i], 0.0))) * weight[i];

        if (texelPos.x+i >= size.x-1) {
            factor = -1;
        }

        col += imageLoad(origTex, texelPos + ivec2(factor * offset[i], 0.0)) * weight[i];
    }

    imageStore(destTex, texelPos, col);
}
