#version 430
#define PI 3.1415926538
#define SIZE 8.0
#define QUALITY 3.0
#define DIRECTIONS 16.0

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, location = 0) writeonly uniform image2D destTex;
layout (rgba8, location = 1) readonly uniform image2D origTex;

uniform float offset[5] = float[](0.0, 1.0, 2.0, 3.0, 4.0);
uniform float weight[5] = float[](0.2270270270, 0.1945945946, 0.1216216216,
                                  0.0540540541, 0.0162162162);

ivec2 abs(ivec2 x) {
    return ivec2(abs(x.x), abs(x.y));
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(origTex);

    vec4 col = imageLoad(origTex, texelPos) * weight[0];

    int factor = 1;

    for (int i = 1; i < 5; i++) {
        col += imageLoad(origTex, abs(texelPos - ivec2(offset[i], 0.0))) * weight[i];

        if (texelPos.x + i >= size.x - 1) {
            factor = -1;
        }

        col += imageLoad(origTex, texelPos + ivec2(factor * offset[i], 0.0)) * weight[i];
    }

    imageStore(destTex, texelPos, col);
}
