struct Uniforms {
    fgColor: vec4<f32>,
    lightDir: vec3<f32>,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
};

@group(0) @binding(0)
var destTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(1)
var sdf: texture_storage_2d<r32float, read>;
@group(0) @binding(2)
var heightField: texture_storage_2d<r32float, read>;
@group(0) @binding(3)
var<uniform> uniforms: Uniforms;

fn loadHeight(pos: vec2<i32>, size: vec2<i32>) -> f32 {
    return textureLoad(heightField, clamp(pos, vec2<i32>(0), size - vec2<i32>(1))).x;
}

fn shade(pos: vec2<i32>, size: vec2<i32>) -> vec4<f32> {
    let h00 = loadHeight(pos + vec2<i32>(-1, -1), size);
    let h10 = loadHeight(pos + vec2<i32>(0, -1), size);
    let h20 = loadHeight(pos + vec2<i32>(1, -1), size);
    let h01 = loadHeight(pos + vec2<i32>(-1, 0), size);
    let h21 = loadHeight(pos + vec2<i32>(1, 0), size);
    let h02 = loadHeight(pos + vec2<i32>(-1, 1), size);
    let h12 = loadHeight(pos + vec2<i32>(0, 1), size);
    let h22 = loadHeight(pos + vec2<i32>(1, 1), size);
    let gx = ((h20 + 2.0 * h21 + h22) - (h00 + 2.0 * h01 + h02)) * 0.125;
    let gy = ((h02 + 2.0 * h12 + h22) - (h00 + 2.0 * h10 + h20)) * 0.125;
    let normal = normalize(vec3<f32>(-gx, -gy, 1.0));
    let rawLight = uniforms.lightDir * vec3<f32>(1.0, -1.0, 1.0);
    let lightLength = max(length(rawLight), 0.0001);
    let light = rawLight / lightLength;
    let ndotl = dot(normal, light);
    let lambert = max(ndotl, 0.0);
    let reflectedZ = max(2.0 * ndotl * normal.z - light.z, 0.0);
    let highlight = select(0.0, pow(reflectedZ, max(uniforms.shininess, 0.0001)), lambert > 0.0 && uniforms.specular > 0.0);
    let illumination = max(uniforms.ambient, 0.0) + max(uniforms.diffuse, 0.0) * lambert;
    let rgb = clamp(uniforms.fgColor.rgb * illumination + max(uniforms.specular, 0.0) * highlight, vec3<f32>(0.0), vec3<f32>(1.0));
    let alpha = clamp(0.5 - textureLoad(sdf, pos).x, 0.0, 1.0);
    return vec4<f32>(rgb, alpha * uniforms.fgColor.a);
}

@compute @workgroup_size(16, 16, 1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(destTex));
    let pos = vec2<i32>(id.xy);
    if (any(pos >= size)) {
        return;
    }
    textureStore(destTex, pos, shade(pos, size));
}
