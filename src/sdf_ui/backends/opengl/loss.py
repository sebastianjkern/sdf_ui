"""OpenGL loss seeding and mask reduction."""

import numpy as np

from .backend import _set
from .glsl import REDUCE

_LOSS = r"""#version 430
layout(local_size_x=64) in;
layout(__FORMAT__,binding=0) readonly uniform image2D prediction;
layout(__FORMAT__,binding=1) readonly uniform image2D targetImage;
layout(__FORMAT__,binding=2) readonly uniform image2D maskImage;
layout(std430,binding=1) buffer Adjoint {uint adjoints[];};
layout(std430,binding=2) writeonly buffer Partials {float partials[];};
layout(std430,binding=4) buffer Status {uint bad;};
uniform ivec2 size;
uniform int resultBase, kind, hasMask;
uniform float scale, delta;
shared float scratch[64];
void addGradient(int index,float value){
 if(value==0.0)return;
 if(isnan(value)||isinf(value)){atomicOr(bad,1u);return;}
 uint previous=atomicCompSwap(adjoints[index],0u,0u),expected;
 do{expected=previous;float next=uintBitsToFloat(expected)+value;
 if(isnan(next)||isinf(next)){atomicOr(bad,1u);return;}
 previous=atomicCompSwap(adjoints[index],expected,floatBitsToUint(next));}while(previous!=expected);
}
void main(){
 int pixel=int(gl_GlobalInvocationID.x),lane=int(gl_LocalInvocationID.x);
 float value=0.0;
 if(pixel<size.x*size.y){
  ivec2 pos=ivec2(pixel%size.x,pixel/size.x);
  vec4 a=imageLoad(prediction,pos),b=imageLoad(targetImage,pos);
  vec4 mask=hasMask!=0 ? imageLoad(maskImage,pos) : vec4(1.0);
  for(int channel=0;channel<__CHANNELS__;channel++){
   if(isnan(mask[channel])||isinf(mask[channel])||mask[channel]<0.0){atomicOr(bad,1u);continue;}
   float error=a[channel]-b[channel],term,gradient;
   if(kind==0){term=error*error;gradient=2.0*error;}
   else if(kind==1){term=abs(error);gradient=sign(error);}
   else if(abs(error)<=delta){term=0.5*error*error;gradient=error;}
   else{term=delta*(abs(error)-0.5*delta);gradient=delta*sign(error);}
   value+=term*scale*mask[channel];
   addGradient(resultBase+pixel*__CHANNELS__+channel,gradient*scale*mask[channel]);
  }
 }
 if(isnan(value)||isinf(value)){atomicOr(bad,1u);value=0.0;}
 scratch[lane]=value;barrier();
 for(int stride=32;stride>0;stride/=2){if(lane<stride)scratch[lane]+=scratch[lane+stride];barrier();}
 if(lane==0)partials[gl_WorkGroupID.x]=scratch[0];
}
"""


def mask_weight_sum(backend, image):
    """Reduce a mask to one scalar on the GPU, without reading its pixels."""
    channels = image.channels
    key = (f"$mask_sum{channels}", False)
    if key not in backend.programs:
        format_ = "r32f" if channels == 1 else "rgba32f"
        source = f"""#version 430
layout(local_size_x=64) in;
layout({format_}, binding=0) readonly uniform image2D maskImage;
layout(std430,binding=2) writeonly buffer Partials {{float partials[];}};
layout(std430,binding=4) buffer Status {{uint bad;}};
shared float scratch[64];
void main() {{
 int i=int(gl_GlobalInvocationID.x), lane=int(gl_LocalInvocationID.x);
 ivec2 size=imageSize(maskImage); float total=0.0;
 if(i<size.x*size.y) {{
  vec4 value=imageLoad(maskImage,ivec2(i%size.x,i/size.x));
  for(int c=0;c<{channels};c++) {{
   if(isnan(value[c]) || isinf(value[c]) || value[c]<0.0) atomicOr(bad,1u);
   else total+=value[c];
  }}
 }}
 scratch[lane]=total; barrier();
 for(int stride=32;stride>0;stride/=2) {{if(lane<stride) scratch[lane]+=scratch[lane+stride]; barrier();}}
 if(lane==0) partials[gl_WorkGroupID.x]=scratch[0];
}}
"""
        backend.programs[key] = backend.ctx.compute_shader(source)
        backend.stats["program_compilations"] += 1
    if ("reduce", True) not in backend.programs:
        backend.programs[("reduce", True)] = backend.ctx.compute_shader(REDUCE)
        backend.stats["program_compilations"] += 1
    groups = (image.size[0] * image.size[1] + 63) // 64
    partials = backend.scratch_buffer("mask_partials", groups * 4)
    result = backend.scratch_buffer("mask_result", 4)
    backend.clear_status()
    image.texture.bind_to_image(0, read=True, write=False)
    partials.bind_to_storage_buffer(2)
    backend.status.bind_to_storage_buffer(4)
    backend.programs[key].run(groups, 1, 1)
    backend.ctx.memory_barrier()
    reducer = backend.programs[("reduce", True)]
    result.bind_to_storage_buffer(3)
    _set(reducer, "groups", groups)
    _set(reducer, "components", 1)
    _set(reducer, "resultOffset", 0)
    reducer.run(1, 1, 1)
    backend.ctx.memory_barrier()
    backend.check_status()
    value = float(np.frombuffer(result.read(), dtype=np.float32)[0])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Mask normalization requires a finite positive sum")
    backend.stats["reduction_dispatches"] += 1
    return value


def seed_pixel_loss(
    backend,
    term,
    prediction,
    target,
    mask,
    adjoints,
    offset,
    values,
    result_index,
    denominator=None,
):
    """Seed an objective root and reduce its scalar value, entirely on GPU."""
    channels = prediction.channels
    key = (f"$loss{channels}", False)
    if key not in backend.programs:
        source = _LOSS.replace(
            "__FORMAT__", "r32f" if channels == 1 else "rgba32f"
        ).replace("__CHANNELS__", str(channels))
        backend.programs[key] = backend.ctx.compute_shader(source)
        backend.stats["program_compilations"] += 1
    if ("reduce", True) not in backend.programs:
        backend.programs[("reduce", True)] = backend.ctx.compute_shader(REDUCE)
        backend.stats["program_compilations"] += 1
    groups = (prediction.size[0] * prediction.size[1] + 63) // 64
    partials = backend.scratch_buffer("loss_partials", groups * 4)
    program = backend.programs[key]
    prediction.texture.bind_to_image(0, read=True, write=False)
    target.texture.bind_to_image(1, read=True, write=False)
    (target if mask is None else mask).texture.bind_to_image(2, read=True, write=False)
    adjoints.bind_to_storage_buffer(1)
    partials.bind_to_storage_buffer(2)
    backend.status.bind_to_storage_buffer(4)
    _set(program, "size", prediction.size)
    _set(program, "resultBase", offset)
    _set(program, "kind", ("mse", "mae", "huber").index(term.kind))
    _set(program, "hasMask", int(mask is not None))
    _set(
        program,
        "scale",
        term.weight
        / (np.prod(prediction.shape) if denominator is None else denominator),
    )
    _set(program, "delta", term.delta)
    program.run(groups, 1, 1)
    backend.ctx.memory_barrier()
    reducer = backend.programs[("reduce", True)]
    values.bind_to_storage_buffer(3)
    _set(reducer, "groups", groups)
    _set(reducer, "components", 1)
    _set(reducer, "resultOffset", result_index)
    reducer.run(1, 1, 1)
    backend.ctx.memory_barrier()
    backend.stats.setdefault("loss_dispatches", 0)
    backend.stats["loss_dispatches"] += 1
    backend.stats["reduction_dispatches"] += 1
