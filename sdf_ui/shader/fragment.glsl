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

// additional point for quadratic bezier curve
uniform vec2 c;

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

float dot2(in vec2 v) { return dot(v, v); }
float cro(in vec2 a, in vec2 b) { return a.x*b.y - a.y*b.x; }

float sdf_quadratic_bezier(in vec2 pos, in vec2 A, in vec2 B, in vec2 C)
{
    vec2 a = B - A;
    vec2 b = A - 2.0*B + C;
    vec2 c = a * 2.0;
    vec2 d = A - pos;

    float kk = 1.0/dot(b, b);
    float kx = kk * dot(a, b);
    float ky = kk * (2.0*dot(a, a)+dot(d, b))/3.0;
    float kz = kk * dot(d, a);

    float res = 0.0;
    float sgn = 0.0;

    float p  = ky - kx*kx;
    float q  = kx*(2.0*kx*kx - 3.0*ky) + kz;
    float p3 = p*p*p;
    float q2 = q*q;
    float h  = q2 + 4.0*p3;

    if (h>=0.0)
    { // 1 root
        h = sqrt(h);
        vec2 x = (vec2(h, -h)-q)/2.0;

        #if 0
        // When p≈0 and p<0, h-q has catastrophic cancelation. So, we do
        // h=√(q²+4p³)=q·√(1+4p³/q²)=q·√(1+w) instead. Now we approximate
        // √ by a linear Taylor expansion into h≈q(1+½w) so that the q's
        // cancel each other in h-q. Expanding and simplifying further we
        // get x=vec2(p³/q,-p³/q-q). And using a second degree Taylor
        // expansion instead: x=vec2(k,-k-q) with k=(1-p³/q²)·p³/q
        if (abs(p)<0.001)
        {
            float k = p3/q;// linear approx
            // float k = (1.0-p3/q2)*p3/q;  // quadratic approx
            x = vec2(k, -k-q);
        }
            #endif

        vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
        float t = clamp(uv.x+uv.y-kx, 0.0, 1.0);
        vec2  q = d+(c+b*t)*t;
        res = dot2(q);
        sgn = cro(c+2.0*b*t, q);
    }
    else
    { // 3 roots
        float z = sqrt(-p);
        float v = acos(q/(p*z*2.0))/3.0;
        float m = cos(v);
        float n = sin(v)*1.732050808;
        vec3  t = clamp(vec3(m+m, -n-m, n-m)*z-kx, 0.0, 1.0);
        vec2  qx=d+(c+b*t.x)*t.x; float dx=dot2(qx), sx = cro(c+2.0*b*t.x, qx);
        vec2  qy=d+(c+b*t.y)*t.y; float dy=dot2(qy), sy = cro(c+2.0*b*t.y, qy);
        if (dx<dy) { res=dx; sgn=sx; } else { res=dy; sgn=sy; }
    }

    return sqrt(res)*sign(sgn) + radius;
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

    #ifdef BEZIER
    float distance = abs(sdf_quadratic_bezier(uv, a, b, c));
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
