#version 460

layout(points) in;
layout(triangle_strip, max_vertices=4) out;

// (min_x, max_y, max_x, min_y)
uniform vec4 bb;
uniform float vertical_stretch;

out vec2 uv;

void main()
{
    float min_x = bb.x;
    float min_y = bb.w;
    float max_x = bb.z;
    float max_y = bb.y;

    // Top right
    gl_Position = vec4(max_x / vertical_stretch, max_y, 0.5, 1.0);
    uv = vec2(max_x, max_y);
    EmitVertex();

    // Top Left
    gl_Position = vec4(min_x / vertical_stretch, max_y, 0.5, 1.0);
    uv = vec2(min_x, max_y);
    EmitVertex();

    // Bottom right
    gl_Position = vec4(max_x / vertical_stretch, min_y, 0.5, 1.0);
    uv = vec2(max_x, min_y);
    EmitVertex();

    // Bottom left
    gl_Position = vec4(min_x / vertical_stretch, min_y, 0.5, 1.0);
    uv = vec2(min_x, min_y);
    EmitVertex();

    EndPrimitive();
}