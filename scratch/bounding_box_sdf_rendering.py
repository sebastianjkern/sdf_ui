import time

import moderngl as mgl

import matplotlib.pyplot as plt

ctx = mgl.create_standalone_context()

n = 50000

size = 1080

width, height = size, size
gw, gh = 16, 16

code = '''
#version 430

#define SMOOTHSTEP_OFFSET 0.0001

layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba8, location=0) writeonly uniform image2D destTex;

uniform vec2 offset;
uniform float radius;

float sdf_circle(in vec2 p, in float r) {
    return length(p)-r;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    float distance = sdf_circle(texelPos - offset, radius);
    
    float fill = smoothstep(-0.75, 0.75+SMOOTHSTEP_OFFSET, distance);
    
    vec4 col1 = vec4(1.0);
    vec4 col2 = vec4(vec3(1.0), 0.0);
    
    vec4 col = mix(col1, col2, fill);
    
    imageStore(destTex, texelPos, col);
}
'''

center = (size/2, size/2)

shader1 = ctx.compute_shader(code)
shader1['destTex'] = 0

code = '''
#version 430

#define SMOOTHSTEP_OFFSET 0.0001

layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba8, location=0) writeonly uniform image2D destTex;

uniform vec2 center;
uniform float radius;

uniform vec2 offset;

float sdf_circle(in vec2 p, in float r) {
    return length(p)-r;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy) + ivec2(offset);

    float distance = sdf_circle(texelPos - center, radius);
    
    float fill = smoothstep(-0.75, 0.75+SMOOTHSTEP_OFFSET, distance);
    
    vec4 col1 = vec4(1.0);
    vec4 col2 = vec4(vec3(1.0), 0.0);
    
    vec4 col = mix(col1, col2, fill);
    
    imageStore(destTex, texelPos, col);
}
'''

shader2 = ctx.compute_shader(code)
shader2['destTex'] = 0

local_size1 = int(width / gw + 0.5), int(height / gh + 0.5), 1

tex = ctx.texture((width, height), 4)

x_radius = []
y_pr_speedup = []
y_th_speedup = []

for radius in range(10, int(size / 2), int(size/25)):
    # Method 1
    shader1['offset'] = center
    shader1['radius'] = radius

    tex.bind_to_image(0, read=False, write=True)

    start = time.time_ns()

    for _ in range(n):
        shader1.run(*local_size1)

    average_time1 = ((time.time_ns() - start) / 1e6) / n

    # print("-" * 15)
    # print(f"Evaluating SDF at {width * height} locations")
    print("Average Time Method 1:", average_time1)

    # Method 2

    rw, rh = int(radius * 2 + 2), int(radius * 2 + 2)

    render_offset = [int(center[0] - radius - 1), int(center[0] - radius - 1)]

    if render_offset[0] < 0:
        rw += render_offset[0]
        render_offset[0] = 0

    if render_offset[1] < 0:
        rh += render_offset[0]
        render_offset[1] = 0

    if render_offset[0] + rw > width:
        rw = width - render_offset[0]

    if render_offset[1] + rh > height:
        rh = height - render_offset[1]

    if render_offset[0] > width:
        # TODO: Discard Rendering, render object is not visible anyway
        pass

    if render_offset[1] > height:
        # TODO: Discard Rendering, render object is not visible anyway
        pass

    local_size2 = int(rw / gw + 0.5), int(rh / gh + 0.5), 1

    shader2['center'] = center
    shader2['radius'] = radius
    shader2['offset'] = tuple(render_offset)

    # print("-" * 15)
    # print(f"Evaluating SDF at {rw * rh} locations")
    # print("Render Size:", (rw, rh))
    # print("Render Offset:", tuple(render_offset))

    start = time.time_ns()

    for _ in range(n):
        shader2.run(*local_size2)

    average_time2 = ((time.time_ns() - start) / 1e6) / n
    print("Average Time Method 2:", average_time2)

    # print("-" * 15)
    # print(f"Theoretical Speedup ~{(width * height) / (rw * rh)} times")
    # print(f"Measured    Speedup ~{average_time1 / average_time2} times")

    x_radius.append(radius)
    y_pr_speedup.append(average_time1 / average_time2)
    y_th_speedup.append((width * height) / (rw * rh))


print(x_radius)
print(y_pr_speedup)
print(y_th_speedup)

# plt.plot(x_radius, y_th_speedup)
plt.plot(x_radius, y_pr_speedup)
plt.show()
