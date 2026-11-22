struct UniformArrayElementI32 {
    @size(16) value: i32,
};

struct D {
    v: f32,
    d: f32,
};

struct D3 {
    x: D,
    y: D,
    z: D,
};

struct D4 {
    x: D,
    y: D,
    z: D,
    w: D,
};

struct Parameters {
    parameterValues: array<f32>,
};

struct AdjointValues {
    adjoints: array<atomic<u32>>,
};

struct Partials {
    partials: array<f32>,
};

struct Status {
    bad: atomic<u32>,
};

const inputChannels: array<i32, 2> = array(1, 1);

struct Uniforms {
    size: vec2<i32>,
    @align(16) inputOffsets: array<UniformArrayElementI32, 2>,
    @align(16) outputOffsets: array<UniformArrayElementI32, 1>,
    uniformBase: i32,
};
@group(0) @binding(2)
var<uniform> _Uniforms: Uniforms;

@group(0) @binding(0)
var source0: texture_storage_2d<r32float, read>;
@group(0) @binding(1)
var source1: texture_storage_2d<r32float, read>;
@group(0) @binding(16)
var<storage, read> parameters: Parameters;
@group(0) @binding(17)
var<storage, read_write> adjointValues: AdjointValues;
@group(0) @binding(18)
var<storage, read_write> partials_2: Partials;
@group(0) @binding(20)
var<storage, read_write> status: Status;
var<private> seedKind: i32 = -1;
var<private> seedIndex: i32 = -1;
var<private> seedInput: i32 = -1;
var<private> seedPixel: i32 = -1;
var<private> seedChannel: i32 = -1;
var<private> accessCount: i32 = 0;
var<private> accessInput: array<i32, 128>;
var<private> accessPixel: array<i32, 128>;
var<private> accessChannel: array<i32, 128>;
var<workgroup> scratch: array<f32, 64>;

fn c(x: f32) -> D {
    return D(x, 0.0);
}

fn selectD(condition: bool, a: D, b: D) -> D {
    return D(select(b.v, a.v, condition), select(b.d, a.d, condition));
}

fn add(a: D, b: D) -> D {
    return D((a.v + b.v), (a.d + b.d));
}

fn sub(a: D, b: D) -> D {
    return D((a.v - b.v), (a.d - b.d));
}

fn neg(a: D) -> D {
    return D(-a.v, -a.d);
}

fn mul(a: D, b: D) -> D {
    return D((a.v * b.v), ((a.d * b.v) + (a.v * b.d)));
}

fn divD(a: D, b: D) -> D {
    return D((a.v / b.v), (((a.d * b.v) - (a.v * b.d)) / (b.v * b.v)));
}

fn sqr(a: D) -> D {
    return mul(a, a);
}

fn sqrtD(a: D) -> D {
    var r: f32 = sqrt(a.v);
    return D(r, select(0.0, ((0.5 * a.d) / r), (r > 0.0)));
}

fn minD(a: D, b: D) -> D {
    return selectD((a.v < b.v), a, selectD((b.v < a.v), b, D(a.v, (0.5 * (a.d + b.d)))));
}

fn maxD(a: D, b: D) -> D {
    return selectD((a.v > b.v), a, selectD((b.v > a.v), b, D(a.v, (0.5 * (a.d + b.d)))));
}

fn clampD(a: D, lo: D, hi: D) -> D {
    return minD(maxD(a, lo), hi);
}

fn powD(a: D, b: D) -> D {
    var v: f32 = pow(a.v, b.v);
    if ((a.v == 0.0))     {
        return D(v, 0.0);
    }
    var derivative: f32 = ((b.v * pow(a.v, (b.v - 1.0))) * a.d);
    if ((b.d != 0.0))     {
        derivative += ((v * log(a.v)) * b.d);
    }
    return D(v, derivative);
}

fn dot3(a: D3, b: D3) -> D {
    return add(add(mul(a.x, b.x), mul(a.y, b.y)), mul(a.z, b.z));
}

fn dp(i: i32) -> D {
    return D(parameters.parameterValues[(_Uniforms.uniformBase + i)], select(0.0, 1.0, ((seedKind == 1) && (seedIndex == i))));
}

fn readInput(which: i32, p: vec2<i32>, channel: i32) -> f32 {
    if (which == 0) {
        return textureLoad(source0, p).x;
    }
    return textureLoad(source1, p).x;
}

fn ld(which: i32, p: vec2<i32>, channel: i32) -> D {
    if (any(p < vec2<i32>(0)) || any(p >= _Uniforms.size)) {
        return c(0.0);
    }
    let texelIndex = p.y * _Uniforms.size.x + p.x;
    if (seedKind == -1) {
        var found = false;
        for (var i = 0; i < accessCount; i += 1) {
            if (accessInput[i] == which && accessPixel[i] == texelIndex && accessChannel[i] == channel) {
                found = true;
            }
        }
        if (!found) {
            if (accessCount >= 128) {
                atomicOr(&status.bad, 2u);
            } else {
                accessInput[accessCount] = which;
                accessPixel[accessCount] = texelIndex;
                accessChannel[accessCount] = channel;
                accessCount += 1;
            }
        }
    }
    return D(readInput(which, p, channel), select(0.0, 1.0, seedKind == 2 && seedInput == which && seedPixel == texelIndex && seedChannel == channel));
}

fn atomicAdjoint(index: i32, value: f32) {
    if ((value == 0.0))     {
        return;
    }
    if ((((bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u) || ((bitcast<u32>(value) & 0x7fffffffu) == 0x7f800000u)))     {
        atomicOr(&status.bad, 1u);
        return;
    }
    let _crosstl_cas_0_ptr = &adjointValues.adjoints[index];
    let _crosstl_cas_0_compare: u32 = 0u;
    let _crosstl_cas_0_value: u32 = 0u;
    var _crosstl_cas_0 = atomicCompareExchangeWeak(_crosstl_cas_0_ptr, _crosstl_cas_0_compare, _crosstl_cas_0_value);
    loop {
        if (_crosstl_cas_0.exchanged || _crosstl_cas_0.old_value != _crosstl_cas_0_compare) {
            break;
        }
        _crosstl_cas_0 = atomicCompareExchangeWeak(_crosstl_cas_0_ptr, _crosstl_cas_0_compare, _crosstl_cas_0_value);
    }
    let _crosstl_cas_0_old: u32 = _crosstl_cas_0.old_value;
    var old: u32 = _crosstl_cas_0_old;
    var expected: u32;
    loop {
        {
            expected = old;
            var next: f32 = (bitcast<f32>(expected) + value);
            if ((((bitcast<u32>(next) & 0x7fffffffu) > 0x7f800000u) || ((bitcast<u32>(next) & 0x7fffffffu) == 0x7f800000u)))             {
                atomicOr(&status.bad, 1u);
                return;
            }
            let _crosstl_cas_1_ptr = &adjointValues.adjoints[index];
            let _crosstl_cas_1_compare: u32 = expected;
            let _crosstl_cas_1_value: u32 = bitcast<u32>(next);
            var _crosstl_cas_1 = atomicCompareExchangeWeak(_crosstl_cas_1_ptr, _crosstl_cas_1_compare, _crosstl_cas_1_value);
            loop {
                if (_crosstl_cas_1.exchanged || _crosstl_cas_1.old_value != _crosstl_cas_1_compare) {
                    break;
                }
                _crosstl_cas_1 = atomicCompareExchangeWeak(_crosstl_cas_1_ptr, _crosstl_cas_1_compare, _crosstl_cas_1_value);
            }
            let _crosstl_cas_1_old: u32 = _crosstl_cas_1.old_value;
            old = _crosstl_cas_1_old;
        }
        continuing {
            break if !(old != expected);
        }
    }
}

fn shadingOutput(color: D4, o: ptr<function, array<D, 8>>) {
    (*o)[0] = color.x;
    (*o)[1] = color.y;
    (*o)[2] = color.z;
    (*o)[3] = color.w;
    for (var i: i32 = 4; (i < 8); i += 1)     {
        (*o)[i] = c(0.0);
    }
}

fn evaluate(pos: vec2<i32>, o: ptr<function, array<D, 8>>) {
    let size = _Uniforms.size;
    let p00 = clamp(pos + vec2<i32>(-1, -1), vec2<i32>(0), size - vec2<i32>(1));
    let p10 = clamp(pos + vec2<i32>(0, -1), vec2<i32>(0), size - vec2<i32>(1));
    let p20 = clamp(pos + vec2<i32>(1, -1), vec2<i32>(0), size - vec2<i32>(1));
    let p01 = clamp(pos + vec2<i32>(-1, 0), vec2<i32>(0), size - vec2<i32>(1));
    let p21 = clamp(pos + vec2<i32>(1, 0), vec2<i32>(0), size - vec2<i32>(1));
    let p02 = clamp(pos + vec2<i32>(-1, 1), vec2<i32>(0), size - vec2<i32>(1));
    let p12 = clamp(pos + vec2<i32>(0, 1), vec2<i32>(0), size - vec2<i32>(1));
    let p22 = clamp(pos + vec2<i32>(1, 1), vec2<i32>(0), size - vec2<i32>(1));
    let h00 = ld(1, p00, 0);
    let h10 = ld(1, p10, 0);
    let h20 = ld(1, p20, 0);
    let h01 = ld(1, p01, 0);
    let h21 = ld(1, p21, 0);
    let h02 = ld(1, p02, 0);
    let h12 = ld(1, p12, 0);
    let h22 = ld(1, p22, 0);
    let gx = mul(sub(add(add(h20, mul(c(2.0), h21)), h22), add(add(h00, mul(c(2.0), h01)), h02)), c(0.125));
    let gy = mul(sub(add(add(h02, mul(c(2.0), h12)), h22), add(add(h00, mul(c(2.0), h10)), h20)), c(0.125));
    let nx = neg(gx);
    let ny = neg(gy);
    let normalLength = sqrtD(add(add(sqr(nx), sqr(ny)), c(1.0)));
    let normal = D3(divD(nx, normalLength), divD(ny, normalLength), divD(c(1.0), normalLength));
    let rawLight = D3(dp(4), neg(dp(5)), dp(6));
    let lightLength = sqrtD(maxD(add(add(sqr(rawLight.x), sqr(rawLight.y)), sqr(rawLight.z)), c(0.00000001)));
    let light = D3(divD(rawLight.x, lightLength), divD(rawLight.y, lightLength), divD(rawLight.z, lightLength));
    let ndotl = dot3(normal, light);
    let lambert = maxD(ndotl, c(0.0));
    let reflectedZ = maxD(sub(mul(mul(c(2.0), ndotl), normal.z), light.z), c(0.0));
    var highlight = c(0.0);
    if (lambert.v > 0.0 && dp(9).v > 0.0 && reflectedZ.v > 0.0) {
        highlight = powD(reflectedZ, maxD(dp(10), c(0.0001)));
    }
    let illumination = add(maxD(dp(7), c(0.0)), mul(maxD(dp(8), c(0.0)), lambert));
    let specular = mul(maxD(dp(9), c(0.0)), highlight);
    let red = clampD(add(mul(dp(0), illumination), specular), c(0.0), c(1.0));
    let green = clampD(add(mul(dp(1), illumination), specular), c(0.0), c(1.0));
    let blue = clampD(add(mul(dp(2), illumination), specular), c(0.0), c(1.0));
    let alpha = clampD(sub(c(0.5), ld(0, pos, 0)), c(0.0), c(1.0));
    shadingOutput(D4(red, green, blue, mul(alpha, dp(3))), o);
}

@compute
@workgroup_size(64, 1, 1)
fn compute_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>, @builtin(local_invocation_id) local_invocation_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    var pixel: i32 = i32(global_invocation_id.x);
    var lane: i32 = i32(local_invocation_id.x);
    var validPixel: bool = (pixel < (_Uniforms.size.x * _Uniforms.size.y));
    var pos: vec2<i32> = vec2<i32>((pixel % _Uniforms.size.x), (pixel / _Uniforms.size.x));
    var upstream: array<f32, 8>;
    for (var i: i32 = 0; (i < 8); i += 1)     {
        upstream[i] = 0.0;
    }
    var outv: array<D, 8>;
    var contributing: bool = false;
    if (validPixel)     {
        upstream[0] = bitcast<f32>(atomicLoad(&adjointValues.adjoints[((_Uniforms.outputOffsets[0].value + (pixel * 4)) + 0)]));
        upstream[1] = bitcast<f32>(atomicLoad(&adjointValues.adjoints[((_Uniforms.outputOffsets[0].value + (pixel * 4)) + 1)]));
        upstream[2] = bitcast<f32>(atomicLoad(&adjointValues.adjoints[((_Uniforms.outputOffsets[0].value + (pixel * 4)) + 2)]));
        upstream[3] = bitcast<f32>(atomicLoad(&adjointValues.adjoints[((_Uniforms.outputOffsets[0].value + (pixel * 4)) + 3)]));
        contributing = ((((upstream[0] != 0.0) || (upstream[1] != 0.0)) || (upstream[2] != 0.0)) || (upstream[3] != 0.0));
        if (contributing)         {
            seedKind = -1;
            evaluate(pos, &outv);
            var count: i32 = accessCount;
            for (var a: i32 = 0; (a < count); a += 1)             {
                seedKind = 2;
                seedInput = accessInput[a];
                seedPixel = accessPixel[a];
                seedChannel = accessChannel[a];
                evaluate(pos, &outv);
                var value: f32 = (((select((outv[0].d * upstream[0]), 0.0, (upstream[0] == 0.0)) + select((outv[1].d * upstream[1]), 0.0, (upstream[1] == 0.0))) + select((outv[2].d * upstream[2]), 0.0, (upstream[2] == 0.0))) + select((outv[3].d * upstream[3]), 0.0, (upstream[3] == 0.0)));
                atomicAdjoint(((_Uniforms.inputOffsets[seedInput].value + (seedPixel * inputChannels[seedInput])) + seedChannel), value);
            }
        }
    }
    for (var parameter: i32 = 0; (parameter < 11); parameter += 1)     {
        var value: f32 = 0.0;
        if ((validPixel && contributing))         {
            seedKind = 1;
            seedIndex = parameter;
            evaluate(pos, &outv);
            value = (((select((outv[0].d * upstream[0]), 0.0, (upstream[0] == 0.0)) + select((outv[1].d * upstream[1]), 0.0, (upstream[1] == 0.0))) + select((outv[2].d * upstream[2]), 0.0, (upstream[2] == 0.0))) + select((outv[3].d * upstream[3]), 0.0, (upstream[3] == 0.0)));
        }
        if ((((bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u) || ((bitcast<u32>(value) & 0x7fffffffu) == 0x7f800000u)))         {
            atomicOr(&status.bad, 1u);
            value = 0.0;
        }
        scratch[lane] = value;
        workgroupBarrier();
        for (var stride: i32 = 32; (stride > 0); stride /= 2)         {
            if ((lane < stride))             {
                scratch[lane] += scratch[(lane + stride)];
            }
            workgroupBarrier();
        }
        if ((lane == 0))         {
            partials_2.partials[((i32(workgroup_id.x) * 11) + parameter)] = scratch[0];
        }
        workgroupBarrier();
    }
}
