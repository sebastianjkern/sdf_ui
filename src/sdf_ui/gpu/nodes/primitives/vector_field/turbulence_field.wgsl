struct Uniforms { frequency: f32, amplitude: f32, octaves: i32, lacunarity: f32, gain: f32, seed: i32, };
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(0) var destTex: texture_storage_2d<rgba32float, write>;
fn hash(p: vec2<i32>, seed: i32) -> f32 {
    var h=u32(p.x)*0x8da6b343u ^ u32(p.y)*0xd8163841u ^ u32(seed)*0xcb1ab31fu;
    h=h^(h>>16u); h=h*0x7feb352du; h=h^(h>>15u); h=h*0x846ca68bu; h=h^(h>>16u);
    return f32(h&0x00ffffffu)/16777215.0;
}
fn valueNoise(p: vec2<f32>, seed: i32) -> f32 {
    let i=vec2<i32>(floor(p)); var f=fract(p); f=f*f*(vec2<f32>(3.0)-2.0*f);
    let a=hash(i,seed); let b=hash(i+vec2<i32>(1,0),seed);
    let c=hash(i+vec2<i32>(0,1),seed); let d=hash(i+vec2<i32>(1,1),seed);
    return mix(mix(a,b,f.x),mix(c,d,f.x),f.y)*2.0-1.0;
}
fn fbm(p: vec2<f32>, seedOffset: i32) -> f32 {
    var sum=0.0; var weight=1.0; var total=0.0; var f=uniforms.frequency;
    for(var i=0; i<8; i++) {
        if(i>=uniforms.octaves) { break; }
        sum+=valueNoise(p*f,uniforms.seed+i+seedOffset)*weight;
        total+=weight; weight*=uniforms.gain; f*=uniforms.lacunarity;
    }
    return sum/max(total,0.0001);
}
@compute @workgroup_size(16,16,1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p=vec2<i32>(id.xy);
    if(any(p>=vec2<i32>(textureDimensions(destTex)))) { return; }
    let xy=vec2<f32>(p);
    textureStore(destTex,p,vec4<f32>(vec2<f32>(fbm(xy,0),fbm(xy+vec2<f32>(37.1,-19.7),101))*uniforms.amplitude,0.0,0.0));
}
