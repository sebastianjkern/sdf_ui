"""Shared WGSL for pointwise HDR primal and pullback kernels."""

from dataclasses import dataclass
from functools import cached_property

_DUAL = """
struct D { v: f32, d: f32, }
alias Pixel = array<D, 4>;
fn k(x: f32) -> D { return D(x, 0.0); }
fn add(a: D, b: D) -> D { return D(a.v+b.v, a.d+b.d); }
fn sub(a: D, b: D) -> D { return D(a.v-b.v, a.d-b.d); }
fn mul(a: D, b: D) -> D { return D(a.v*b.v, a.d*b.v+a.v*b.d); }
fn div(a: D, b: D) -> D { return D(a.v/b.v, (a.d-a.v/b.v*b.d)/b.v); }
fn positive(a: D) -> D { if a.v>0.0 { return a; } return k(0.0); }
fn clip(a: D) -> D {
    return D(clamp(a.v, 0.0, 1.0), select(0.0, a.d, a.v>0.0 && a.v<1.0));
}
fn power(a: D, n: f32) -> D {
    let y = pow(a.v, n);
    return D(y, n*pow(a.v, n-1.0)*a.d);
}
fn scale2(a: D) -> D { let x = exp2(a.v); return D(x, x*0.69314718056*a.d); }
fn mix_d(a: D, b: D, t: D) -> D { return add(a, mul(sub(b,a),t)); }
fn decode(a: D) -> D {
    if a.v <= 0.04045 { return div(a,k(12.92)); }
    return power(div(add(a,k(0.055)),k(1.055)),2.4);
}
fn encode(a: D) -> D {
    if a.v <= 0.0031308 { return mul(a,k(12.92)); }
    return sub(mul(k(1.055),power(a,1.0/2.4)),k(0.055));
}
fn luminance(p: Pixel) -> D {
    return add(add(mul(k(0.2126),p[0]),mul(k(0.7152),p[1])),mul(k(0.0722),p[2]));
}
fn bayer(pos: vec2<i32>) -> f32 {
    let pattern = array<f32,16>(0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5);
    return ((pattern[(pos.y%4)*4+pos.x%4]+0.5)/16.0-0.5)/255.0;
}
"""
_TYPES = {"float": "f32", "vec4": "vec4<f32>"}


@dataclass(frozen=True)
class PointwiseKernel:
    contract: object
    body: str
    mixed: str = ""
    helpers: str = ""

    def _inputs(self, start, *, display=False):
        for i, arg in enumerate(self.contract.inputs):
            fmt = "r32float" if arg.kind == "r32f" else "rgba32float"
            if display and self.mixed == "decode":
                fmt = "rgba8unorm"
            yield (
                f"@group(0) @binding({start + i}) "
                f"var {arg.name}: texture_storage_2d<{fmt}, read>;"
            )

    def _evaluate(self):
        return (
            "fn evaluate(pos: vec2<i32>, ins: array<Pixel, "
            f"{max(1, len(self.contract.inputs))}>, u: array<D, {max(1, self.components)}>) "
            "-> Pixel {\n" + self.body + "\n}\n"
        )

    @property
    def components(self):
        return sum(4 if arg.kind == "vec4" else 1 for arg in self.contract.parameters)

    def forward(self, *, display=False):
        contract = self.contract
        fmt = "rgba8unorm" if display and self.mixed == "finish" else "rgba32float"
        fields = "\n".join(
            f"{arg.name}: {_TYPES[arg.kind]}," for arg in contract.parameters
        )
        uniforms = (
            f"struct ForwardUniforms {{ {fields} }}\n"
            f"@group(0) @binding({len(contract.inputs) + 1}) var<uniform> params: ForwardUniforms;"
            if fields
            else ""
        )
        loads = "\n".join(
            f"let in{i} = textureLoad({arg.name}, pos);"
            f" for (var j=0; j<{4 if arg.shape else 1}; j++) {{ ins[{i}][j]=k(in{i}[j]); }}"
            for i, arg in enumerate(contract.inputs)
        )
        args = []
        for arg in contract.parameters:
            args.extend(
                f"k(params.{arg.name}{suffix})"
                for suffix in (
                    ("",) if arg.kind == "float" else ("[0]", "[1]", "[2]", "[3]")
                )
            )
        args = ", ".join(args or ["k(0.0)"])
        output = contract.outputs[0].name
        return "\n".join(
            (
                *self._inputs(1, display=display),
                uniforms,
                _DUAL,
                self.helpers,
                self._evaluate(),
                f"@group(0) @binding(0) var {output}: texture_storage_2d<{fmt}, write>;",
                f"""
@compute @workgroup_size(16, 16, 1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let pos = vec2<i32>(id.xy);
    if any(pos >= vec2<i32>(textureDimensions({output}))) {{ return; }}
    var ins: array<Pixel, {max(1, len(contract.inputs))}>;
    {loads}
    let result = evaluate(pos, ins, array<D, {max(1, self.components)}>({args}));
    textureStore({output}, pos, vec4<f32>(result[0].v,result[1].v,result[2].v,result[3].v));
}}
""",
            )
        )

    @cached_property
    def backward(self):
        count = len(self.contract.inputs)
        input_count = max(1, count)
        components = max(1, self.components)
        fetch = "\n".join(
            f"let in{i} = textureLoad({arg.name}, pos);"
            f" for (var j=0; j<{4 if arg.shape else 1}; j++) {{ ins[{i}][j]=k(in{i}[j]); }}"
            for i, arg in enumerate(self.contract.inputs)
        )
        gradient = "\n".join(
            f"""
        for (var j=0; j<{channels}; j++) {{
            ins[{i}][j].d = 1.0;
            let gradient = pull(evaluate(pos, ins, u), upstream);
            ins[{i}][j].d = 0.0;
            accumulate(_Uniforms.inputOffsets[{i}].value+pixel*{channels}+j, gradient);
        }}
"""
            for i, arg in enumerate(self.contract.inputs)
            for channels in (4 if arg.shape else 1,)
        )
        return "\n".join(
            (
                *self._inputs(0),
                _DUAL,
                self.helpers,
                self._evaluate(),
                f"""
struct Offset {{ @size(16) value: i32, }}
struct Uniforms {{
    size: vec2<i32>,
    @align(16) inputOffsets: array<Offset, {input_count}>,
    @align(16) outputOffsets: array<Offset, 1>,
}}
@group(0) @binding({count})
var<uniform> _Uniforms: Uniforms;
@group(0) @binding(16) var<storage, read> params: array<f32>;
@group(0) @binding(17) var<storage, read_write> adjoints: array<atomic<u32>>;
@group(0) @binding(18) var<storage, read_write> partials: array<f32>;
@group(0) @binding(20) var<storage, read_write> status: atomic<u32>;
var<workgroup> scratch: array<f32, 64>;
fn finite_or_zero(x: f32) -> f32 {{
    if ((bitcast<u32>(x)&0x7fffffffu)>=0x7f800000u) {{
        atomicOr(&status,1u); return 0.0;
    }}
    return x;
}}
fn accumulate(index: i32, value: f32) {{
    let x = finite_or_zero(value);
    if x==0.0 {{ return; }}
    var old = atomicLoad(&adjoints[index]);
    loop {{
        let update = atomicCompareExchangeWeak(&adjoints[index],old,bitcast<u32>(bitcast<f32>(old)+x));
        if update.exchanged {{ break; }}
        old = update.old_value;
    }}
}}
fn pull(output: Pixel, upstream: vec4<f32>) -> f32 {{
    var result=0.0;
    for (var i=0; i<4; i++) {{
        if upstream[i]!=0.0 {{ result+=output[i].d*upstream[i]; }}
    }}
    return result;
}}
@compute @workgroup_size(64, 1, 1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>) {{
    let pixel=i32(id.x);
    let pos=vec2<i32>(pixel%_Uniforms.size.x,pixel/_Uniforms.size.x);
    var ins: array<Pixel, {input_count}>;
    var u: array<D, {components}>;
    for (var i=0; i<{self.components}; i++) {{ u[i]=k(params[i]); }}
    var upstream=vec4<f32>(0.0);
    if pos.y<_Uniforms.size.y {{
        let offset=_Uniforms.outputOffsets[0].value+pixel*4;
        for (var i=0; i<4; i++) {{ upstream[i]=bitcast<f32>(atomicLoad(&adjoints[offset+i])); }}
    }}
    let contributes=any(upstream!=vec4<f32>(0.0));
    if contributes {{
        {fetch}
        {gradient}
    }}
    for (var i=0; i<{self.components}; i++) {{
        var g=0.0;
        if contributes {{
            u[i].d=1.0;
            g=finite_or_zero(pull(evaluate(pos,ins,u),upstream));
            u[i].d=0.0;
        }}
        scratch[local.x]=g;
        workgroupBarrier();
        for (var step=32u; step>0u; step/=2u) {{
            if local.x<step {{ scratch[local.x]+=scratch[local.x+step]; }}
            workgroupBarrier();
        }}
        if local.x==0u {{ partials[group.x*{components}u+u32(i)]=scratch[0]; }}
        workgroupBarrier();
    }}
}}
""",
            )
        )
