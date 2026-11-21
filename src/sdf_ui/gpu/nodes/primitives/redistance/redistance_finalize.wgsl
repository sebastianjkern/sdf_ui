@group(0) @binding(0)
var source_field: texture_storage_2d<r32float, read>;
@group(0) @binding(1)
var seeds: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2)
var distance_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(source_field);
    if (id.x >= dimensions.x || id.y >= dimensions.y) {
        return;
    }

    let point = vec2<i32>(id.xy);
    let original = textureLoad(source_field, point).x;
    let seed = textureLoad(seeds, point);
    if (seed.w < 0.5) {
        textureStore(distance_out, point, vec4<f32>(original, 0.0, 0.0, 0.0));
        return;
    }

    let delta = vec2<f32>(point) - seed.xy;
    let distance = length(delta);
    let signed_distance = select(distance, -distance, original < 0.0);
    textureStore(distance_out, point, vec4<f32>(signed_distance, 0.0, 0.0, 0.0));
}
