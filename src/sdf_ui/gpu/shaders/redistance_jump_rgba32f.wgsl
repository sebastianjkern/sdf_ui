struct Parameters {
    jump: u32,
};

@group(0) @binding(0)
var seeds_in: texture_storage_2d<rgba32float, read>;
@group(0) @binding(1)
var seeds_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2)
var<uniform> parameters: Parameters;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(seeds_in);
    if (id.x >= dimensions.x || id.y >= dimensions.y) {
        return;
    }

    let point = vec2<i32>(id.xy);
    let point_f = vec2<f32>(point);
    var best = textureLoad(seeds_in, point);
    var best_distance = 1.0e30;
    if (best.w > 0.5) {
        let delta = point_f - best.xy;
        best_distance = dot(delta, delta);
    }

    let jump = i32(parameters.jump);
    for (var oy = -1; oy <= 1; oy += 1) {
        for (var ox = -1; ox <= 1; ox += 1) {
            let candidate_point = point + vec2<i32>(ox, oy) * jump;
            if (candidate_point.x < 0 || candidate_point.y < 0 ||
                candidate_point.x >= i32(dimensions.x) ||
                candidate_point.y >= i32(dimensions.y)) {
                continue;
            }
            let candidate = textureLoad(seeds_in, candidate_point);
            if (candidate.w < 0.5) {
                continue;
            }
            let delta = point_f - candidate.xy;
            let distance_squared = dot(delta, delta);
            if (distance_squared < best_distance) {
                best = candidate;
                best_distance = distance_squared;
            }
        }
    }

    textureStore(seeds_out, point, best);
}
