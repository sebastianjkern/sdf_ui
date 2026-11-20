"""Deterministic, smoothly interpolated grain in canvas pixel coordinates."""


def reference_noise(context, size):
    import numpy as np

    def sample(x, y):
        value = x * np.uint32(0x9E3779B9) + y * np.uint32(0x85EBCA6B)
        value ^= value >> np.uint32(16)
        value *= np.uint32(0x7FEB352D)
        value ^= value >> np.uint32(15)
        value *= np.uint32(0x846CA68B)
        value ^= value >> np.uint32(16)
        return (value & np.uint32(0x00FFFFFF)).astype(np.float64) / 8388608.0 - 1

    x, y = context.x / size, context.y / size
    ix, iy = np.floor(x.data), np.floor(y.data)
    wx, wy = x - ix, y - iy
    wx = wx * wx * (3 - 2 * wx)
    wy = wy * wy * (3 - 2 * wy)
    ix, iy = ix.astype(np.uint32), iy.astype(np.uint32)
    left, right = sample(ix, iy), sample(ix + 1, iy)
    bottom = wx * (right - left) + left
    left, right = sample(ix, iy + 1), sample(ix + 1, iy + 1)
    top = wx * (right - left) + left
    return bottom + wy * (top - bottom)


WGSL = """
fn grain_hash(p: vec2<i32>) -> D {
    let xy = bitcast<vec2<u32>>(p);
    var value = xy.x * 0x9E3779B9u + xy.y * 0x85EBCA6Bu;
    value ^= value >> 16u;
    value *= 0x7FEB352Du;
    value ^= value >> 15u;
    value *= 0x846CA68Bu;
    value ^= value >> 16u;
    return k(f32(value & 0x00FFFFFFu) / 8388608.0 - 1.0);
}
fn grain_weight(f: D) -> D {
    return mul(mul(f,f),sub(k(3.0),mul(k(2.0),f)));
}
fn grain_noise(pos: vec2<i32>, size: D) -> D {
    let x = div(k(f32(pos.x)),size);
    let y = div(k(f32(pos.y)),size);
    let cell = vec2<i32>(i32(floor(x.v)),i32(floor(y.v)));
    let wx = grain_weight(sub(x,k(f32(cell.x))));
    let wy = grain_weight(sub(y,k(f32(cell.y))));
    let bottom = mix_d(grain_hash(cell),grain_hash(cell+vec2<i32>(1,0)),wx);
    let top = mix_d(grain_hash(cell+vec2<i32>(0,1)),grain_hash(cell+vec2<i32>(1,1)),wx);
    return mix_d(bottom,top,wy);
}
"""
