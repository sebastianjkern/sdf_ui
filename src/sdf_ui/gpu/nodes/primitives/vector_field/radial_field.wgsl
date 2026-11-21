struct Uniforms { center: vec2<f32>, radius: f32, strength: f32, };
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(0) var destTex: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(16,16,1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p=vec2<i32>(id.xy);
    if(any(p>=vec2<i32>(textureDimensions(destTex)))) { return; }
    let delta=vec2<f32>(p)-uniforms.center;
    let attenuation=exp(-dot(delta,delta)/(2.0*uniforms.radius*uniforms.radius));
    textureStore(destTex,p,vec4<f32>(delta/uniforms.radius*uniforms.strength*attenuation,0.0,0.0));
}
