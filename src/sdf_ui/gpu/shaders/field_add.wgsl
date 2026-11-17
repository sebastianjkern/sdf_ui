@group(0) @binding(0) var destTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(1) var field0: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var field1: texture_storage_2d<rgba32float, read>;
@compute @workgroup_size(16,16,1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p=vec2<i32>(id.xy);
    if(any(p>=vec2<i32>(textureDimensions(destTex)))) { return; }
    textureStore(destTex,p,textureLoad(field0,p)+textureLoad(field1,p));
}
