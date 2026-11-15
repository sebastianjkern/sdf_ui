@group(0) @binding(0)
var source_field: texture_storage_2d<r32float, read>;
@group(0) @binding(1)
var seeds_out: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(source_field);
    if (id.x >= dimensions.x || id.y >= dimensions.y) {
        return;
    }

    let point = vec2<i32>(id.xy);
    let value = textureLoad(source_field, point).x;
    var nearest = vec2<f32>(-1.0, -1.0);
    var best_distance = 1.0e30;

    if (value == 0.0) {
        nearest = vec2<f32>(point);
        best_distance = 0.0;
    }

    let directions = array<vec2<i32>, 4>(
        vec2<i32>(1, 0), vec2<i32>(-1, 0),
        vec2<i32>(0, 1), vec2<i32>(0, -1),
    );
    for (var index = 0; index < 4; index += 1) {
        let direction = directions[index];
        let neighbor = point + direction;
        if (neighbor.x < 0 || neighbor.y < 0 ||
            neighbor.x >= i32(dimensions.x) || neighbor.y >= i32(dimensions.y)) {
            continue;
        }
        let other = textureLoad(source_field, neighbor).x;
        if ((value < 0.0) == (other < 0.0)) {
            continue;
        }

        let fraction = abs(value) / max(abs(value) + abs(other), 1.0e-20);
        let crossing = vec2<f32>(point) + vec2<f32>(direction) * fraction;
        let delta = vec2<f32>(point) - crossing;
        let distance_squared = dot(delta, delta);
        if (distance_squared < best_distance) {
            nearest = crossing;
            best_distance = distance_squared;
        }
    }

    let valid = select(0.0, 1.0, best_distance < 1.0e29);
    textureStore(seeds_out, point, vec4<f32>(nearest, 0.0, valid));
}
