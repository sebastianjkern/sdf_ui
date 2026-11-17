#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(r32f, binding = 0) writeonly uniform image2D destTex;
layout(r32f, binding = 1) readonly uniform image2D sdf0;
uniform float columns;
uniform float rows;
uniform vec2 spacing;
uniform vec2 center;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(destTex);
    if (any(greaterThanEqual(pixel, size))) return;
    vec2 counts = vec2(columns, rows);
    vec2 anchor = floor((vec2(pixel) - center) / spacing + (counts - 1.0) * 0.5 + 0.5);
    float distance = float(max(size.x, size.y));
    ivec2 sourceSize = imageSize(sdf0);
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 cell = anchor + vec2(x, y);
            if (cell.x < 0.0 || cell.y < 0.0 || cell.x >= columns || cell.y >= rows) continue;
            vec2 offset = (cell - (counts - 1.0) * 0.5) * spacing;
            ivec2 source = ivec2(round(vec2(pixel) - offset));
            if (any(lessThan(source, ivec2(0))) || any(greaterThanEqual(source, sourceSize))) continue;
            distance = min(distance, imageLoad(sdf0, source).r);
        }
    }
    imageStore(destTex, pixel, vec4(distance, 0.0, 0.0, 0.0));
}
