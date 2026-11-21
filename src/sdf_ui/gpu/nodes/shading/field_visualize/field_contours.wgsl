struct Uniforms { spacing: f32, line_width: f32, scale: f32, };
@group(0) @binding(0) var destTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var fieldTex: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@compute @workgroup_size(16,16,1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p=vec2<i32>(id.xy); if(any(p>=vec2<i32>(textureDimensions(destTex)))) { return; }
    let value=length(textureLoad(fieldTex,p).rg)*uniforms.scale;
    let band=abs(value+uniforms.spacing*0.5-uniforms.spacing*floor((value+uniforms.spacing*0.5)/uniforms.spacing)-uniforms.spacing*0.5);
    let low=max(uniforms.line_width-0.75,0.0); let high=uniforms.line_width+0.75;
    let t=clamp((band-low)/(high-low),0.0,1.0); let line=1.0-t*t*(3.0-2.0*t);
    let color=mix(vec3<f32>(0.035,0.055,0.095),vec3<f32>(0.27,0.88,0.92),line);
    textureStore(destTex,p,vec4<f32>(color,1.0));
}
