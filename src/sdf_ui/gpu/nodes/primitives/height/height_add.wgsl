@group(0) @binding(0) var destTex: texture_storage_2d<r32float, write>;
@group(0) @binding(1) var height0: texture_storage_2d<r32float, read>;
@group(0) @binding(2) var height1: texture_storage_2d<r32float, read>;
var<private> pixelSize: vec2<i32>;
@compute @workgroup_size(16, 16, 1)
fn compute_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let p = vec2<i32>(global_invocation_id.xy);
    pixelSize = vec2<i32>(textureDimensions(destTex));
    if (any(p >= pixelSize)) { return; }
    let h = textureLoad(height0, p).x + textureLoad(height1, p).x;
    textureStore(destTex, p, vec4<f32>(h, 0.0, 0.0, 0.0));
}
