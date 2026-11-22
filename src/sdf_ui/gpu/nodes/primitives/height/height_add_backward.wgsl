struct UniformArrayElementI32 { @size(16) value: i32, };
struct AdjointValues { adjoints: array<atomic<u32>>, };
struct Status { bad: atomic<u32>, };
struct Uniforms {
    size: vec2<i32>,
    @align(16) outputOffsets: array<UniformArrayElementI32, 1>,
    @align(16) inputOffsets: array<UniformArrayElementI32, 2>,
};
@group(0) @binding(2) var<uniform> _Uniforms: Uniforms;
@group(0) @binding(17) var<storage, read_write> adjointValues: AdjointValues;
@group(0) @binding(20) var<storage, read_write> status: Status;

fn atomicAdjoint(index: i32, value: f32) {
    if (value == 0.0) { return; }
    var old = atomicLoad(&adjointValues.adjoints[index]);
    loop {
        let expected = old;
        let next = bitcast<u32>(bitcast<f32>(expected) + value);
        if ((next & 0x7fffffffu) >= 0x7f800000u) {
            atomicOr(&status.bad, 1u);
            return;
        }
        let result = atomicCompareExchangeWeak(&adjointValues.adjoints[index], expected, next);
        if (result.exchanged) { break; }
        old = result.old_value;
    }
}

@compute @workgroup_size(64, 1, 1)
fn compute_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = i32(gid.x);
    if (pixel < _Uniforms.size.x * _Uniforms.size.y) {
        let upstream = bitcast<f32>(atomicLoad(&adjointValues.adjoints[_Uniforms.outputOffsets[0].value + pixel]));
        atomicAdjoint(_Uniforms.inputOffsets[0].value + pixel, upstream);
        atomicAdjoint(_Uniforms.inputOffsets[1].value + pixel, upstream);
    }
}
