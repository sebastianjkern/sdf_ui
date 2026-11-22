struct UniformArrayElementI32 { @size(16) value: i32, };
struct Parameters { parameterValues: array<f32>, };
struct AdjointValues { adjoints: array<atomic<u32>>, };
struct Partials { partials: array<f32>, };
struct Status { bad: atomic<u32>, };
struct Uniforms {
    size: vec2<i32>,
    @align(16) outputOffsets: array<UniformArrayElementI32, 1>,
    @align(16) inputOffsets: array<UniformArrayElementI32, 1>,
    uniformBase: i32,
};
@group(0) @binding(2) var<uniform> _Uniforms: Uniforms;
@group(0) @binding(16) var<storage, read> parameters: Parameters;
@group(0) @binding(17) var<storage, read_write> adjointValues: AdjointValues;
@group(0) @binding(18) var<storage, read_write> partials_2: Partials;
@group(0) @binding(20) var<storage, read_write> status: Status;
var<workgroup> scratch: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn compute_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let pixel = i32(gid.x);
    let lane = i32(lid.x);
    var gradient: array<f32, 5>;
    for (var i = 0; i < 5; i += 1) { gradient[i] = 0.0; }
    if (pixel < _Uniforms.size.x * _Uniforms.size.y) {
        let pos = vec2<i32>(pixel % _Uniforms.size.x, pixel / _Uniforms.size.x);
        let upstream = bitcast<f32>(atomicLoad(&adjointValues.adjoints[_Uniforms.outputOffsets[0].value + pixel]));
        let dx = f32(pos.x) - parameters.parameterValues[_Uniforms.uniformBase];
        let dy = f32(pos.y) - parameters.parameterValues[_Uniforms.uniformBase + 1];
        let sx = parameters.parameterValues[_Uniforms.uniformBase + 2];
        let sy = parameters.parameterValues[_Uniforms.uniformBase + 3];
        let amplitude = parameters.parameterValues[_Uniforms.uniformBase + 4];
        let nx = dx / sx;
        let ny = dy / sy;
        let e = exp(-0.5 * (nx * nx + ny * ny));
        if (e > 0.0) {
            let weight = upstream * amplitude * e;
            gradient[0] = weight * nx / sx;
            gradient[1] = weight * ny / sy;
            gradient[2] = weight * nx * nx / sx;
            gradient[3] = weight * ny * ny / sy;
        }
        gradient[4] = upstream * e;
    }
    for (var i = 0; i < 5; i += 1) {
        scratch[lane] = gradient[i];
        workgroupBarrier();
        for (var stride = 32; stride > 0; stride /= 2) {
            if (lane < stride) { scratch[lane] += scratch[lane + stride]; }
            workgroupBarrier();
        }
        if (lane == 0) { partials_2.partials[i32(wid.x) * 5 + i] = scratch[0]; }
        workgroupBarrier();
    }
}
