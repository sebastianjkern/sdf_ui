@group(0) @binding(0)
var destination: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1)
var source: texture_2d<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let position = vec2<i32>(invocation.xy);
    if (any(invocation.xy >= textureDimensions(destination))) {
        return;
    }
    textureStore(destination, position, textureLoad(source, position, 0));
}
