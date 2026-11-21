struct Node {
    low: vec2<f32>,
    high: vec2<f32>,
    a: vec2<f32>,
    b: vec2<f32>,
    escape: u32,
    leaf: u32,
};

struct Path {
    radius: f32,
    count: u32,
    nodes: array<Node>,
};

@group(0) @binding(0) var output: texture_storage_2d<r32float, write>;
@group(0) @binding(1) var<storage, read> path: Path;

fn segment_distance_sq(p: vec2<f32>, node: Node) -> f32 {
    let edge = node.b - node.a;
    let relative = p - node.a;
    let length_sq = dot(edge, edge);
    var t = 0.0;
    if length_sq > 0.0 {
        t = clamp(dot(relative, edge) / length_sq, 0.0, 1.0);
    }
    let delta = relative - edge * t;
    return dot(delta, delta);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(output)) {
        return;
    }
    let p = vec2<f32>(id.xy);
    var best = 1e32;
    if path.count > 0u {
        // Start with distances to a few actual segments to prune distant boxes.
        best = min(segment_distance_sq(p, path.nodes[0]), segment_distance_sq(p, path.nodes[path.count / 2u]));
        best = min(best, segment_distance_sq(p, path.nodes[path.count - 1u]));
    }
    var index = 0u;
    while index < path.count {
        let node = path.nodes[index];
        let delta = max(max(node.low - p, p - node.high), vec2<f32>(0.0));
        if dot(delta, delta) >= best {
            index = node.escape;
        } else {
            if node.leaf != 0u {
                best = min(best, segment_distance_sq(p, node));
            }
            index += 1u;
        }
    }
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(sqrt(best) - path.radius, 0.0, 0.0, 0.0));
}
