struct Uniforms { scale: f32, magnitude: f32, };
@group(0) @binding(0) var destTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var fieldTex: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
fn hsv(h: f32, s: f32, v: f32) -> vec3<f32> {
    let p=abs(fract(vec3<f32>(h)+vec3<f32>(0.0,2.0/3.0,1.0/3.0))*6.0-vec3<f32>(3.0));
    return v*mix(vec3<f32>(1.0),clamp(p-vec3<f32>(1.0),vec3<f32>(0.0),vec3<f32>(1.0)),s);
}
@compute @workgroup_size(16,16,1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p=vec2<i32>(id.xy); if(any(p>=vec2<i32>(textureDimensions(destTex)))) { return; }
    let v=textureLoad(fieldTex,p).rg*uniforms.scale;
    let hue=atan2(v.y,v.x)/6.28318530718+0.5;
    let brightness=1.0-exp(-length(v)*uniforms.magnitude);
    textureStore(destTex,p,vec4<f32>(hsv(hue,0.78,brightness),1.0));
}
