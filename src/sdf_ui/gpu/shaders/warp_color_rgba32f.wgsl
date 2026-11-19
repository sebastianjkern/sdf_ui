struct Uniforms { strength: f32, steps: i32, };
@group(0) @binding(0) var destTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(1) var sourceTex: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var fieldTex: texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
fn fieldAt(p0: vec2<f32>) -> vec2<f32> {
    let size=vec2<i32>(textureDimensions(fieldTex)); let p=clamp(p0,vec2<f32>(0.0),vec2<f32>(size-1));
    let a=vec2<i32>(floor(p)); let b=min(a+vec2<i32>(1),size-1); let f=fract(p);
    let v00=textureLoad(fieldTex,a).rg; let v10=textureLoad(fieldTex,vec2<i32>(b.x,a.y)).rg;
    let v01=textureLoad(fieldTex,vec2<i32>(a.x,b.y)).rg; let v11=textureLoad(fieldTex,b).rg;
    return mix(mix(v00,v10,f.x),mix(v01,v11,f.x),f.y);
}
fn sourceAt(p: vec2<f32>) -> vec4<f32> {
    let size=vec2<i32>(textureDimensions(sourceTex));
    if(any(p<vec2<f32>(0.0))||any(p>vec2<f32>(size-1))) { return vec4<f32>(0.0); }
    let q=clamp(p,vec2<f32>(0.0),vec2<f32>(size-1)); let a=vec2<i32>(floor(q)); let b=min(a+vec2<i32>(1),size-1); let f=fract(q);
    let v00=textureLoad(sourceTex,a); let v10=textureLoad(sourceTex,vec2<i32>(b.x,a.y));
    let v01=textureLoad(sourceTex,vec2<i32>(a.x,b.y)); let v11=textureLoad(sourceTex,b);
    return mix(mix(v00,v10,f.x),mix(v01,v11,f.x),f.y);
}
@compute @workgroup_size(16,16,1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p=vec2<i32>(id.xy); if(any(p>=vec2<i32>(textureDimensions(destTex)))) { return; }
    var position=vec2<f32>(p); let stepSize=uniforms.strength/f32(uniforms.steps);
    for(var i=0;i<uniforms.steps;i++) { position-=fieldAt(position)*stepSize; }
    textureStore(destTex,p,sourceAt(position));
}
