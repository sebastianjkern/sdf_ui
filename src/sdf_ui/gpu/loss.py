"""WebGPU pixel-loss seeding and mask normalization reductions."""

from __future__ import annotations

import struct

import numpy as np

_MASK = """
struct Status { bad: atomic<u32>, };
fn finite_value(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
@group(0) @binding(0) var mask_image: texture_2d<f32>;
@group(0) @binding(17) var<storage, read_write> partials: array<f32>;
@group(0) @binding(20) var<storage, read_write> status: Status;
var<workgroup> sums: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3<u32>) {
    let size = textureDimensions(mask_image);
    let index = gid.x;
    var value = 0.0;
    if (index < size.x * size.y) {
        let pixel = textureLoad(mask_image, vec2<i32>(i32(index % size.x), i32(index / size.x)), 0);
        for (var channel = 0u; channel < __CHANNELS__u; channel += 1u) {
            let item = pixel[channel];
            if (!finite_value(item) || item < 0.0) { atomicOr(&status.bad, 1u); }
            else { value += item; }
        }
    }
    sums[lane] = value;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (lane < stride) { sums[lane] += sums[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u) { partials[group.x] = sums[0]; }
}
"""

_LOSS = """
struct Config {
    size: vec2<i32>,
    result_base: u32,
    kind: u32,
    has_mask: u32,
    scale: f32,
    delta: f32,
    _padding: u32,
};
struct Status { bad: atomic<u32>, };
fn finite_value(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
@group(0) @binding(0) var prediction: texture_2d<f32>;
@group(0) @binding(1) var target_image: texture_2d<f32>;
@group(0) @binding(2) var mask_image: texture_2d<f32>;
@group(0) @binding(3) var<uniform> config: Config;
@group(0) @binding(16) var<storage, read_write> adjoints: array<atomic<u32>>;
@group(0) @binding(17) var<storage, read_write> partials: array<f32>;
@group(0) @binding(20) var<storage, read_write> status: Status;
var<workgroup> sums: array<f32, 64>;
fn add_gradient(index: u32, value: f32) {
    if (value == 0.0) { return; }
    if (!finite_value(value)) { atomicOr(&status.bad, 1u); return; }
    var previous = atomicLoad(&adjoints[index]);
    loop {
        let next = bitcast<f32>(previous) + value;
        if (!finite_value(next)) { atomicOr(&status.bad, 1u); return; }
        let result = atomicCompareExchangeWeak(&adjoints[index], previous, bitcast<u32>(next));
        if (result.exchanged) { return; }
        previous = result.old_value;
    }
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3<u32>) {
    let index = gid.x;
    let pixel_count = u32(config.size.x * config.size.y);
    var loss = 0.0;
    if (index < pixel_count) {
        let pos = vec2<i32>(i32(index % u32(config.size.x)), i32(index / u32(config.size.x)));
        let a = textureLoad(prediction, pos, 0);
        let b = textureLoad(target_image, pos, 0);
        var mask = vec4<f32>(1.0);
        if (config.has_mask != 0u) { mask = textureLoad(mask_image, pos, 0); }
        for (var channel = 0u; channel < __CHANNELS__u; channel += 1u) {
            let weight = mask[channel];
            if (!finite_value(weight) || weight < 0.0) { atomicOr(&status.bad, 1u); continue; }
            let error = a[channel] - b[channel];
            var value = 0.0;
            var gradient = 0.0;
            if (config.kind == 0u) { value = error * error; gradient = 2.0 * error; }
            else if (config.kind == 1u) { value = abs(error); gradient = sign(error); }
            else if (abs(error) <= config.delta) { value = 0.5 * error * error; gradient = error; }
            else { value = config.delta * (abs(error) - 0.5 * config.delta); gradient = config.delta * sign(error); }
            loss += value * config.scale * weight;
            add_gradient(config.result_base + index * __CHANNELS__u + channel, gradient * config.scale * weight);
        }
    }
    if (!finite_value(loss)) { atomicOr(&status.bad, 1u); loss = 0.0; }
    sums[lane] = loss;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (lane < stride) { sums[lane] += sums[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u) { partials[group.x] = sums[0]; }
}
"""


def _groups(image):
    return (image.size[0] * image.size[1] + 63) // 64


def _partial_sum(buffer, count):
    values = np.frombuffer(buffer.read(count * 4), dtype=np.float32)
    return float(np.sum(values, dtype=np.float64))


def mask_weight_sum(backend, image):
    """Validate and sum a mask on the GPU, reading back only workgroup sums."""
    backend._check()
    groups = _groups(image)
    partials = backend.scratch_buffer("mask_partials", groups * 4)
    backend.clear_status()
    source = _MASK.replace("__CHANNELS__", str(image.channels))
    backend.canvas.run_compute(
        source, {0: image, 17: partials, 20: backend.status},
        group_count=(groups, 1, 1), label="mask weight sum",
    )
    backend.stats["reduction_dispatches"] += 1
    backend.check_status()
    total = _partial_sum(partials, groups)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Mask normalization requires a finite positive sum")
    return total


def seed_pixel_loss(
    backend, term, prediction, target, mask, adjoints, offset, values,
    result_index, denominator=None,
):
    """Seed pixel adjoints and reduce the scalar objective on the GPU."""
    backend._check()
    if prediction.size != target.size or prediction.channels != target.channels:
        raise ValueError("Prediction and target image shapes differ")
    if mask is not None and (mask.size != prediction.size or mask.channels not in (1, 4)):
        raise ValueError("Loss mask shape differs from prediction")
    groups = _groups(prediction)
    partials = backend.scratch_buffer("loss_partials", groups * 4)
    config_buffer = backend._uniform_buffer(
        "pixel loss config",
        struct.pack(
            "<2iIIIffI", prediction.size[0], prediction.size[1], int(offset),
            ("mse", "mae", "huber").index(term.kind), int(mask is not None),
            float(term.weight / (np.prod(prediction.shape) if denominator is None else denominator)),
            float(term.delta), 0,
        ),
    )
    backend.clear_status()
    source = _LOSS.replace("__CHANNELS__", str(prediction.channels))
    backend.canvas.run_compute(
        source,
        {
            0: prediction, 1: target, 2: target if mask is None else mask,
            3: config_buffer, 16: adjoints, 17: partials, 20: backend.status,
        },
        group_count=(groups, 1, 1), label="pixel loss",
    )
    backend.stats["loss_dispatches"] += 1
    backend.check_status()
    loss = _partial_sum(partials, groups)
    if not np.isfinite(loss):
        raise FloatingPointError("WebGPU pixel loss is nonfinite")
    values.write(struct.pack("<f", loss), int(result_index) * 4)
