struct Uniforms { vector: vec2<f32>, };
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(0) var destTex: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(16,16,1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p=vec2<i32>(id.xy);
    if(any(p>=vec2<i32>(textureDimensions(destTex)))) { return; }
    textureStore(destTex,p,vec4<f32>(uniforms.vector,0.0,0.0));
}
