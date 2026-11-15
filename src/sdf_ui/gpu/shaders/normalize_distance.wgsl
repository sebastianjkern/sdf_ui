struct Parameters {
    epsilon: f32,
    band: f32,
};

@group(0) @binding(0)
var dest_tex: texture_storage_2d<r32float, write>;
@group(0) @binding(1)
var source_field: texture_storage_2d<r32float, read>;
@group(0) @binding(2)
var<uniform> parameters: Parameters;

@compute @workgroup_size(16, 16, 1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(dest_tex);
    if (id.x >= dimensions.x || id.y >= dimensions.y) {
        return;
    }

    let point = vec2<i32>(id.xy);
    let max_point = vec2<i32>(dimensions) - vec2<i32>(1, 1);
    let value = textureLoad(source_field, point).x;
    let left = textureLoad(source_field, max(point - vec2<i32>(1, 0), vec2<i32>(0, 0))).x;
    let right = textureLoad(source_field, min(point + vec2<i32>(1, 0), max_point)).x;
    let up = textureLoad(source_field, max(point - vec2<i32>(0, 1), vec2<i32>(0, 0))).x;
    let down = textureLoad(source_field, min(point + vec2<i32>(0, 1), max_point)).x;

    let dx_divisor = select(2.0, 1.0, point.x == 0 || point.x == max_point.x);
    let dy_divisor = select(2.0, 1.0, point.y == 0 || point.y == max_point.y);
    let gradient = length(vec2<f32>((right - left) / dx_divisor, (down - up) / dy_divisor));
    let inverse_gradient = select(1.0, 1.0 / gradient, gradient > parameters.epsilon);
    let proximity = clamp(1.0 - abs(value) / parameters.band, 0.0, 1.0);
    let weight = proximity * proximity * (3.0 - 2.0 * proximity);
    let corrected = value * (1.0 + weight * (inverse_gradient - 1.0));
    textureStore(dest_tex, point, vec4<f32>(corrected, 0.0, 0.0, 0.0));
}
