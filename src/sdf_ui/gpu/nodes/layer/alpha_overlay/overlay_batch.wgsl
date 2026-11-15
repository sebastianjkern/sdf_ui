const BASE_IS_LAB: bool = false;
{{TOP_FLAGS}}

{{OUTPUT_BINDINGS}}
@group(0) @binding({{BASE_BINDING}})
var baseTex: texture_storage_2d<rgba8unorm, read>;
{{TOP_BINDINGS}}

fn inverse_lab_curve(value: f32) -> f32 {
    if (value > 0.206897) {
        return value * value * value;
    }
    return (value - (16.0 / 116.0)) / 7.787;
}

fn gamma_rgb(value: f32) -> f32 {
    if (value > 0.0031308) {
        return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
    }
    return 12.92 * value;
}

fn lab_to_rgb(lab: vec3<f32>) -> vec3<f32> {
    let y = (100.0 * lab.x + 16.0) / 116.0;
    let x = 254.0 * (lab.y - 0.5) / 500.0 + y;
    let z = y - 254.0 * (lab.z - 0.5) / 200.0;
    let xyz = vec3<f32>(
        0.95047 * inverse_lab_curve(x),
        inverse_lab_curve(y),
        1.08883 * inverse_lab_curve(z),
    );
    return vec3<f32>(
        gamma_rgb(dot(xyz, vec3<f32>(3.2406, -1.5372, -0.4986))),
        gamma_rgb(dot(xyz, vec3<f32>(-0.9689, 1.8758, 0.0415))),
        gamma_rgb(dot(xyz, vec3<f32>(0.0557, -0.204, 1.057))),
    );
}

fn quantize_rgba8(color: vec4<f32>) -> vec4<f32> {
    return round(clamp(color, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0) / 255.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let position = vec2<i32>(invocation.xy);
    let size = textureDimensions(baseTex);
    if (any(position >= vec2<i32>(size))) {
        return;
    }

    var accum = textureLoad(baseTex, position);
    if (BASE_IS_LAB) {
        accum = quantize_rgba8(vec4<f32>(lab_to_rgb(accum.rgb), accum.a));
    }
    {{STEPS}}
}
