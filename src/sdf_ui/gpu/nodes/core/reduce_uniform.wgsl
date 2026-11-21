struct ReduceConfig {
    group_count: u32,
    component_count: u32,
    result_offset: u32,
    _padding: u32,
};

struct Status {
    bad: atomic<u32>,
};

fn finite_value(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

@group(0) @binding(0)
var<storage, read> partials: array<f32>;
@group(0) @binding(1)
var<storage, read_write> gradients: array<atomic<u32>>;
@group(0) @binding(2)
var<uniform> config: ReduceConfig;
@group(0) @binding(3)
var<storage, read_write> status: Status;

fn add_gradient(index: u32, value: f32) {
    var previous = atomicLoad(&gradients[index]);
    loop {
        let next = bitcast<f32>(previous) + value;
        if (!finite_value(next)) {
            atomicOr(&status.bad, 1u);
            return;
        }
        let result = atomicCompareExchangeWeak(
            &gradients[index], previous, bitcast<u32>(next)
        );
        if (result.exchanged) {
            return;
        }
        previous = result.old_value;
    }
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let component = invocation.x;
    if (component >= config.component_count) {
        return;
    }
    var total = 0.0;
    for (var group = 0u; group < config.group_count; group += 1u) {
        total += partials[group * config.component_count + component];
    }
    if (!finite_value(total)) {
        atomicOr(&status.bad, 1u);
        return;
    }
    add_gradient(config.result_offset + component, total);
}
