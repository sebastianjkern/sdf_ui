"""Native graph passes for generated spatial shaders."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256

from .nodes.base import ImageBinding, ShaderFile, ShaderInterface, Uniform
from .nodes.differentiation import ShaderArgument, ShaderDifferentiation
from .spatial import Compiler

_SCALAR_KINDS = {"sdf", "height"}
_FORMATS = {"sdf": "r32float", "height": "r32float", "field": "rgba32float", "color": "rgba32float"}
_PORTS = {"sdf": "r32f", "height": "r32f", "field": "rgba32float", "color": "rgba8"}
_STATE = """
var<private> world: vec2<f32>;
var<private> cell: vec2<f32>;
var<private> basis: mat2x2<f32>;
fn reset_coordinates(position: vec2<f32>) {
    world=position;
    cell=world;
    basis=mat2x2<f32>(vec2<f32>(1,0),vec2<f32>(0,1));
}
"""


@dataclass(frozen=True)
class SpatialKernel:
    """Reusable WGSL assembled once per plan.

    The primal executes arbitrary callable shader fragments. The pullback
    uses local centered differences through the same program and accumulates
    contributions into shared source texels on the GPU.
    """

    functions: str
    target: str
    source_kinds: tuple[str, ...]
    result_kind: str
    values: int

    def _samplers(self, start, formats, *, trace=False):
        code = []
        raw_cases = []
        for i, fmt in enumerate(formats):
            code.append(f"""
@group(0) @binding({start + i})
var tex_{i}: texture_storage_2d<{fmt}, read>;
fn load_{i}(p: vec2<i32>) -> vec4<f32> {{
    let value=textureLoad(tex_{i},p);
    {"return observe(value," + str(i) + ",p);" if trace else "return value;"}
}}
fn sample_{i}(p: vec2<f32>) -> vec4<f32> {{
    let q=clamp(p,vec2<f32>(0.0),vec2<f32>(canvas_extent())-1.0);
    let lo=vec2<i32>(floor(q));
    let hi=min(lo+1,vec2<i32>(canvas_extent())-1);
    let t=fract(q);
    return mix(mix(load_{i}(lo),load_{i}(vec2<i32>(hi.x,lo.y)),t.x),
               mix(load_{i}(vec2<i32>(lo.x,hi.y)),load_{i}(hi),t.x),t.y);
}}
""")
            raw_cases.append(f"case {i}: {{ return textureLoad(tex_{i},p); }}")
        if trace:
            code.append(
                "fn source_value(which: i32, p: vec2<i32>) -> vec4<f32> {"
                "switch which {"
                + "\n".join(raw_cases)
                + " default: { return vec4<f32>(0.0); } } }"
            )
        return "\n".join(code)

    def forward(self, *, display=False):
        formats = tuple(
            "rgba8unorm" if display and kind == "color" else _FORMATS[kind]
            for kind in self.source_kinds
        )
        target = (
            "rgba8unorm"
            if display and self.result_kind == "color"
            else _FORMATS[self.result_kind]
        )
        return self.forward_formats(target, formats)

    def forward_formats(self, output, inputs):
        return f"""
struct SampleValue {{ @size(16) value: f32, }}
struct SpatialValues {{ values: array<SampleValue,{self.values}>, }}
@group(0) @binding(0) var dest: texture_storage_2d<{output},write>;
@group(0) @binding({len(inputs) + 1}) var<uniform> uniforms: SpatialValues;
fn canvas_extent() -> vec2<u32> {{ return textureDimensions(dest); }}
fn spatial_value(i: i32) -> f32 {{ return uniforms.values[i].value; }}
{_STATE}
{self._samplers(1, inputs)}
{self.functions}
@compute @workgroup_size(16,16,1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {{
    if any(id.xy>=canvas_extent()) {{ return; }}
    reset_coordinates(vec2<f32>(id.xy));
    textureStore(dest,vec2<i32>(id.xy),{self.target}(world));
}}
"""

    @cached_property
    def backward(self):
        maps = len(self.source_kinds)
        formats = tuple(_FORMATS[kind] for kind in self.source_kinds)
        channels = tuple(1 if kind in _SCALAR_KINDS else 4 for kind in self.source_kinds)
        output_channels = 1 if self.result_kind in _SCALAR_KINDS else 4
        capacity = 128
        return f"""
struct Offset {{ @size(16) value: i32, }}
struct Uniforms {{
    size: vec2<i32>,
    @align(16) inputOffsets: array<Offset,{max(1, maps)}>,
    @align(16) outputOffsets: array<Offset,1>,
}}
@group(0) @binding({maps})
var<uniform> _Uniforms: Uniforms;
@group(0) @binding(16) var<storage,read> parameters: array<f32>;
@group(0) @binding(17) var<storage,read_write> adjoints: array<atomic<u32>>;
@group(0) @binding(18) var<storage,read_write> partials: array<f32>;
@group(0) @binding(20) var<storage,read_write> status: atomic<u32>;
const counts=array<i32,{max(1, maps)}>({", ".join(map(str, channels or (1,)))});
var<workgroup> scratch: array<f32,64>;
var<private> tracing: bool;
var<private> reads: array<vec3<i32>,{capacity}>;
var<private> read_count: i32;
var<private> seed: vec4<i32>;
var<private> perturbation: f32;
fn canvas_extent() -> vec2<u32> {{ return vec2<u32>(_Uniforms.size); }}
fn spatial_value(i: i32) -> f32 {{
    return parameters[i] + select(0.0, perturbation, seed.x == -1 && seed.y == i);
}}
fn observe(value: vec4<f32>, which: i32, p: vec2<i32>) -> vec4<f32> {{
    let address=vec3<i32>(which,p);
    if tracing {{
        for (var i=0;i<read_count;i++) {{
            if all(reads[i]==address) {{ return value; }}
        }}
        if read_count=={capacity} {{ atomicOr(&status,2u); return value; }}
        reads[read_count]=address;
        read_count+=1;
        return value;
    }}
    if all(seed.xyz==address) {{
        var shifted=value;
        shifted[seed.w]+=perturbation;
        return shifted;
    }}
    return value;
}}
fn get_adjoint(index: i32) -> f32 {{ return bitcast<f32>(atomicLoad(&adjoints[index])); }}
fn finite_or_zero(value: f32) -> f32 {{
    if (bitcast<u32>(value)&0x7fffffffu)>=0x7f800000u {{
        atomicOr(&status,1u);
        return 0.0;
    }}
    return value;
}}
fn accumulate(index: i32, value: f32) {{
    let x=finite_or_zero(value);
    if x==0.0 {{ return; }}
    var old=atomicLoad(&adjoints[index]);
    loop {{
        let result=atomicCompareExchangeWeak(&adjoints[index],old,bitcast<u32>(bitcast<f32>(old)+x));
        if result.exchanged {{ break; }}
        old=result.old_value;
    }}
}}
{_STATE}
{self._samplers(0, formats, trace=True)}
{self.functions}
fn evaluate_at(point: vec2<f32>) -> vec4<f32> {{
    reset_coordinates(point);
    return {self.target}(point);
}}
fn slope(point: vec2<f32>, upstream: vec4<f32>, h: f32) -> f32 {{
    perturbation=h;
    let positive=evaluate_at(point);
    perturbation=-h;
    let negative=evaluate_at(point);
    return dot((positive-negative)/(2.0*h),upstream);
}}
fn step_size(value: f32) -> f32 {{
    return max(0.001,abs(value)*0.00005);
}}
@compute @workgroup_size(64,1,1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>,
                @builtin(local_invocation_id) local: vec3<u32>,
                @builtin(workgroup_id) group: vec3<u32>) {{
    let pixel=i32(id.x);
    let integer_point=vec2<i32>(pixel%_Uniforms.size.x,pixel/_Uniforms.size.x);
    let point=vec2<f32>(integer_point);
    var upstream=vec4<f32>(0.0);
    if integer_point.y<_Uniforms.size.y {{
        let index=_Uniforms.outputOffsets[0].value+pixel*{output_channels};
        for (var c=0;c<{output_channels};c++) {{ upstream[c]=get_adjoint(index+c); }}
    }}
    let contributes=any(upstream!=vec4<f32>(0.0));
    if contributes && {maps}>0 {{
        read_count=0; tracing=true; seed=vec4<i32>(-2);
        let unused=evaluate_at(point);
        tracing=false;
        for (var i=0;i<read_count;i++) {{
            let address=reads[i];
            let which=address.x;
            let p=address.yz;
            let src=source_value(which,p);
            for (var channel=0;channel<counts[which];channel++) {{
                seed=vec4<i32>(address,channel);
                let g=slope(point,upstream,step_size(src[channel]));
                let source_pixel=p.y*_Uniforms.size.x+p.x;
                let index=_Uniforms.inputOffsets[which].value+source_pixel*counts[which]+channel;
                accumulate(index,g);
            }}
        }}
    }}
    for (var i=0;i<{self.values};i++) {{
        var g=0.0;
        if contributes {{
            seed=vec4<i32>(-1,i,0,0);
            g=finite_or_zero(slope(point,upstream,step_size(parameters[i])));
        }}
        scratch[local.x]=g;
        workgroupBarrier();
        for (var stride=32u;stride>0u;stride/=2u) {{
            if local.x<stride {{ scratch[local.x]+=scratch[local.x+stride]; }}
            workgroupBarrier();
        }}
        if local.x==0u {{ partials[group.x*{self.values}u+u32(i)]=scratch[0]; }}
        workgroupBarrier();
    }}
}}
"""


def _spatial_argument(name, kind):
    return ShaderArgument(
        name, () if kind in _SCALAR_KINDS else (4,), _PORTS[kind], port_kind=kind,
    )


def _shader_file(name, path, kernel, node, constants, kinds):
    arguments = tuple(
        _spatial_argument(f"tex_{index}", kind)
        for index, kind in enumerate(kinds)
    )
    contract = ShaderDifferentiation(
        name, path, arguments,
        (ShaderArgument("values", (len(constants),), "float"),),
        (_spatial_argument("dest", node.kind),),
        sampling="spatial",
        notes="GPU pullback uses centered differences through the fused parameter program.",
    )
    interface = ShaderInterface(
        ImageBinding("dest", 0, write=True),
        tuple(ImageBinding(arg.name, index + 1, read=True)
              for index, arg in enumerate(arguments)),
        (Uniform("values", f"float[{len(constants)}]"),),
    )
    return ShaderFile(
        name, path, derivative_contract=contract, interface=interface, wgsl=kernel
    ), arguments


def emit_spatial(builder, node):
    """Lower the procedural template with its map textures as graph inputs."""
    from ..learning.render import _tensor

    compiler = Compiler()
    target = compiler.evaluate(node)
    constants = tuple(compiler.constants or (0.0,))
    maps = tuple(item[0] for item in compiler.maps)
    kinds = tuple(item.kind for item in maps)
    kernel = SpatialKernel(
        "\n".join(compiler.functions), target, kinds, node.kind, len(constants)
    )
    name = "spatial_" + sha256(repr(kernel).encode()).hexdigest()
    shader, arguments = _shader_file(
        name, f"generated/{name}.wgsl", kernel, node, constants, kinds
    )
    builder.structure.append((name, builder.signature(constants)))
    ports = {argument.name: builder.node(dependency)
             for argument, dependency in zip(arguments, maps)}

    def uniforms(resolve):
        return {"values": _tensor(tuple(resolve(value) for value in constants))}

    return builder.emit(shader, ports, uniforms, getattr(node, "mode", None))[0]
