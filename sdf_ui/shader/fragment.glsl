in vec2 uv;

// Universal compile time constants
#define PI 3.1415926538
#define SMOOTHSTEP_OFFSET 0.0000000001

// Universal runtime constants
uniform float antialiasing_distance;
uniform float elevation;

uniform vec2 center;
uniform vec2 size;
uniform vec4 corner_radius;
uniform float radius;
uniform float vertical_stretch;
uniform vec4 obj_col;
uniform vec3 shadow_col;

// Line stuff
uniform vec2 a;
uniform vec2 b;

// Universal
uniform vec4 outline_color;
uniform vec4 outline_r;

float sdf_circle(in vec2 uv, in float radius, in vec2 center)
{
    return length(uv - center) - radius;
}

float sdf_roundbox(in vec2 uv, in vec2 size, in vec4 corner_radius)
{
    corner_radius.xy = mix(corner_radius.xy, corner_radius.zw, smoothstep(0, SMOOTHSTEP_OFFSET, uv.x));
    corner_radius.x = mix(corner_radius.x, corner_radius.y, smoothstep(0, SMOOTHSTEP_OFFSET, uv.y));
    vec2 q = abs(uv)-size+corner_radius.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - corner_radius.x;
}

float sdf_linesegment(in vec2 uv, in vec2 a, in vec2 b, in float radius)
{
    vec2 ba = b-a;
    vec2 pa = uv-a;
    float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
    return length(pa-h*ba) - radius;
}

vec3 rgb2xyz(vec3 c)
{
    vec3 tmp;
    tmp.x = (c.r > 0.04045) ? pow((c.r + 0.055) / 1.055, 2.4) : c.r / 12.92;
    tmp.y = (c.g > 0.04045) ? pow((c.g + 0.055) / 1.055, 2.4) : c.g / 12.92,
    tmp.z = (c.b > 0.04045) ? pow((c.b + 0.055) / 1.055, 2.4) : c.b / 12.92;
    const mat3x3 mat = mat3x3(
    0.4124, 0.3576, 0.1805,
    0.2126, 0.7152, 0.0722,
    0.0193, 0.1192, 0.9505
    );
    return 100.0 * tmp * mat;
}

vec3 xyz2lab(vec3 c)
{
    vec3 n = c / vec3(95.047, 100, 108.883);
    vec3 v;
    v.x = (n.x > 0.008856) ? pow(n.x, 1.0 / 3.0) : (7.787 * n.x) + (16.0 / 116.0);
    v.y = (n.y > 0.008856) ? pow(n.y, 1.0 / 3.0) : (7.787 * n.y) + (16.0 / 116.0);
    v.z = (n.z > 0.008856) ? pow(n.z, 1.0 / 3.0) : (7.787 * n.z) + (16.0 / 116.0);
    return vec3((116.0 * v.y) - 16.0, 500.0 * (v.x - v.y), 200.0 * (v.y - v.z));
}

vec3 rgb2lab(vec3 c)
{
    vec3 lab = xyz2lab(rgb2xyz(c));
    return vec3(lab.x / 100.0, 0.5 + 0.5 * (lab.y / 127.0), 0.5 + 0.5 * (lab.z / 127.0));
}

vec3 lab2xyz(vec3 c)
{
    float fy = (c.x + 16.0) / 116.0;
    float fx = c.y / 500.0 + fy;
    float fz = fy - c.z / 200.0;
    return vec3(
    95.047 * ((fx > 0.206897) ? fx * fx * fx : (fx - 16.0 / 116.0) / 7.787),
    100.000 * ((fy > 0.206897) ? fy * fy * fy : (fy - 16.0 / 116.0) / 7.787),
    108.883 * ((fz > 0.206897) ? fz * fz * fz : (fz - 16.0 / 116.0) / 7.787)
    );
}

vec3 xyz2rgb(vec3 c)
{
    const mat3x3 mat = mat3x3(
    3.2406, -1.5372, -0.4986,
    -0.9689, 1.8758, 0.0415,
    0.0557, -0.2040, 1.0570
    );
    vec3 v = c / 100.0 * mat;
    vec3 r;
    r.x = (v.r > 0.0031308) ? ((1.055 * pow(v.r, (1.0 / 2.4))) - 0.055) : 12.92 * v.r;
    r.y = (v.g > 0.0031308) ? ((1.055 * pow(v.g, (1.0 / 2.4))) - 0.055) : 12.92 * v.g;
    r.z = (v.b > 0.0031308) ? ((1.055 * pow(v.b, (1.0 / 2.4))) - 0.055) : 12.92 * v.b;
    return r;
}

vec3 lab2rgb(vec3 c)
{
    return xyz2rgb(lab2xyz(vec3(100.0 * c.x, 2.0 * 127.0 * (c.y - 0.5), 2.0 * 127.0 * (c.z - 0.5))));
}

void main()
{
    #ifdef CIRCLE
    float distance = sdf_circle(uv, radius, center);
    #endif

    #ifdef RECT
    float distance = sdf_roundbox(uv - center, size, corner_radius);
    #endif

    #ifdef LINE
    float distance = sdf_linesegment(uv, a, b, radius);
    #endif

    // Antialising
    float smooth_d = smoothstep(0, antialiasing_distance + SMOOTHSTEP_OFFSET, distance);

    // Shadow rendering
    float shadow_angle = PI - atan(elevation / (distance * distance));

    // Clip elevation to 0 or 1 depending on the activation of the shadow rendering
    float bgr_factor = smoothstep(0, SMOOTHSTEP_OFFSET, elevation);

    // Convert colors to lab color space
    // for visually more pleasing color interpolation
    vec4 obj_col_lab = vec4(rgb2lab(obj_col.xyz), obj_col.w);
    vec3 sha_col_lab = rgb2lab(shadow_col);

    // Interpolate between object color and shadow color if elevation is non zero
    vec4 bgr_col = vec4(mix(obj_col_lab.xyz, sha_col_lab.xyz, bgr_factor), 1-shadow_angle/PI);

    vec4 out_lab = mix(obj_col_lab, bgr_col, smooth_d);
    vec4 out_rgb = vec4(lab2rgb(out_lab.xyz), out_lab.w);

    gl_FragColor = out_rgb;
}
