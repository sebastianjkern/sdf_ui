struct Uniforms {
    center: vec2<f32>,
    sigma: vec2<f32>,
    amplitude: f32,
};
@group(0) @binding(1) var<uniform> _Uniforms: Uniforms;
@group(0) @binding(0) var destTex: texture_storage_2d<r32float, write>;
var<private> pixelSize: vec2<i32>;
@compute @workgroup_size(16, 16, 1)
fn compute_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let pos = vec2<i32>(global_invocation_id.xy);
    pixelSize = vec2<i32>(textureDimensions(destTex));
    if (any(pos >= pixelSize)) { return; }
    let p = (vec2<f32>(pos) - _Uniforms.center) / _Uniforms.sigma;
    let h = _Uniforms.amplitude * exp(-0.5 * dot(p, p));
    textureStore(destTex, pos, vec4<f32>(h, 0.0, 0.0, 0.0));
}
