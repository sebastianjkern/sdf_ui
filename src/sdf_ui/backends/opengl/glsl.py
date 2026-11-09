"""Scalar dual arithmetic shared by native GLSL derivative kernels."""

from ...core.dual_glsl import PRELUDE
from ..base import shape_size


def derivative_source(contract, body):
    """Generate a GPU local-VJP pass with scatter and workgroup reductions."""
    channels = [int(shape_size(argument.shape)) for argument in contract.inputs]
    parameter_sizes = [
        int(shape_size(argument.shape)) for argument in contract.parameters
    ]
    count = sum(parameter_sizes)
    output_components = [
        (i, channel)
        for i, arg in enumerate(contract.outputs)
        for channel in range(int(shape_size(arg.shape)))
    ]
    declarations, reads, macros = [], [], []
    for index, argument in enumerate(contract.inputs):
        format_ = "r32f" if not argument.shape else "rgba32f"
        declarations.append(
            f"layout({format_},binding={index}) readonly uniform image2D source{index};"
        )
        reads.append(f"if(which=={index}) return imageLoad(source{index},p)[channel];")
    offset = 0
    for argument, n in zip(contract.parameters, parameter_sizes):
        name = "U_" + argument.name
        if len(argument.shape) == 1 and argument.kind == "float":
            macros.append(f"#define {name}(i) dp({offset}+(i))")
        elif n == 1:
            macros.append(f"#define {name} dp({offset})")
        else:
            macros.append(
                f"#define {name} D{n}("
                + ",".join(f"dp({offset + i})" for i in range(n))
                + ")"
            )
        offset += n
    zero_indices = []
    offset = 0
    for argument, n in zip(contract.parameters, parameter_sizes):
        if argument.name in contract.zero_gradient_parameters:
            zero_indices.extend(range(offset, offset + n))
        offset += n
    zero_check = " || ".join(f"i=={i}" for i in zero_indices) or "false"
    zero_parameter = " || ".join(f"parameter=={i}" for i in zero_indices) or "false"
    seeds = "\n".join(
        f"upstream[{j}]=uintBitsToFloat(adjoints[outputOffsets[{i}]+pixel*{shape_size(contract.outputs[i].shape)}+{channel}]);"
        for j, (i, channel) in enumerate(output_components)
    )
    weighted = (
        "+".join(
            f"(upstream[{i}]==0.0?0.0:outv[{i}].d*upstream[{i}])"
            for i in range(len(output_components))
        )
        or "0.0"
    )
    input_channels = ",".join(str(c) for c in channels) if channels else "1"
    any_upstream = (
        " || ".join(f"upstream[{i}]!=0.0" for i in range(len(output_components)))
        or "false"
    )
    return (
        "#version 430\n"
        + PRELUDE
        + "\n"
        + "\n".join(declarations)
        + f"""
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Parameters {{ float parameterValues[]; }};
layout(std430,binding=1) buffer AdjointValues {{ uint adjoints[]; }};
layout(std430,binding=2) writeonly buffer Partials {{ float partials[]; }};
layout(std430,binding=4) buffer Status {{ uint bad; }};
uniform ivec2 size;
uniform int inputOffsets[{max(len(channels), 1)}];
uniform int outputOffsets[{max(len(contract.outputs), 1)}];
uniform int uniformBase;
const int inputChannels[{max(len(channels), 1)}]=int[]({input_channels});
int seedKind=-1, seedIndex=-1, seedInput=-1, seedPixel=-1, seedChannel=-1;
int accessCount=0;
int accessInput[128], accessPixel[128], accessChannel[128];
shared float scratch[64];
D dp(int i) {{ return D(parameterValues[uniformBase+i],seedKind==1 && seedIndex==i && !({zero_check})?1.0:0.0); }}
float readInput(int which,ivec2 p,int channel) {{ {"".join(reads)} return 0.0; }}
D ld(int which,ivec2 p,int channel) {{
    if(any(lessThan(p,ivec2(0))) || any(greaterThanEqual(p,size))) return c(0.0);
    int texelIndex=p.y*size.x+p.x;
    if(seedKind==-1) {{
        bool found=false;
        for(int i=0;i<accessCount;i++) if(accessInput[i]==which && accessPixel[i]==texelIndex && accessChannel[i]==channel) found=true;
        if(!found) {{
            if(accessCount>=128) atomicOr(bad,2u);
            else {{ accessInput[accessCount]=which;accessPixel[accessCount]=texelIndex;accessChannel[accessCount]=channel;accessCount++; }}
        }}
    }}
    return D(readInput(which,p,channel),seedKind==2 && seedInput==which && seedPixel==texelIndex && seedChannel==channel?1.0:0.0);
}}
void atomicAdjoint(int index,float value) {{
    if(value==0.0) return;
    if(isnan(value)||isinf(value)) {{atomicOr(bad,1u);return;}}
    uint old=atomicCompSwap(adjoints[index],0u,0u), expected;
    do {{
        expected=old;
        float next=uintBitsToFloat(expected)+value;
        if(isnan(next)||isinf(next)) {{atomicOr(bad,1u);return;}}
        old=atomicCompSwap(adjoints[index],expected,floatBitsToUint(next));
    }} while(old!=expected);
}}
"""
        + "\n".join(macros)
        + "\n"
        + body
        + f"""
void main() {{
    int pixel=int(gl_GlobalInvocationID.x), lane=int(gl_LocalInvocationID.x);
    bool validPixel=pixel<size.x*size.y;
    ivec2 pos=ivec2(pixel%size.x,pixel/size.x);
    float upstream[8];
    for(int i=0;i<8;i++) upstream[i]=0.0;
    D outv[8];
    bool contributing=false;
    if(validPixel) {{
        {seeds}
        contributing={any_upstream};
        if(contributing) {{
            seedKind=-1;evaluate(pos,outv);
            int count=accessCount;
            for(int a=0;a<count;a++) {{
                seedKind=2;seedInput=accessInput[a];seedPixel=accessPixel[a];seedChannel=accessChannel[a];
                evaluate(pos,outv);
                float value={weighted};
                atomicAdjoint(inputOffsets[seedInput]+seedPixel*inputChannels[seedInput]+seedChannel,value);
            }}
        }}
    }}
    for(int parameter=0;parameter<{count};parameter++) {{
        if({zero_parameter}) {{
            if(lane==0)partials[int(gl_WorkGroupID.x)*{max(count, 1)}+parameter]=0.0;
            continue;
        }}
        float value=0.0;
        if(validPixel && contributing) {{seedKind=1;seedIndex=parameter;evaluate(pos,outv);value={weighted};}}
        if(isnan(value)||isinf(value)) {{atomicOr(bad,1u);value=0.0;}}
        scratch[lane]=value;barrier();
        for(int stride=32;stride>0;stride/=2) {{if(lane<stride)scratch[lane]+=scratch[lane+stride];barrier();}}
        if(lane==0)partials[int(gl_WorkGroupID.x)*{max(count, 1)}+parameter]=scratch[0];
        barrier();
    }}
}}
"""
    )


REDUCE = r"""#version 430
layout(local_size_x=64) in;
layout(std430,binding=2) readonly buffer Partials {float partials[];};
layout(std430,binding=3) buffer Results {float results[];};
layout(std430,binding=4) buffer Status {uint bad;};
uniform int groups, components, resultOffset;
shared float scratch[64];
void main(){
 int component=int(gl_WorkGroupID.x),lane=int(gl_LocalInvocationID.x);
 float value=0.0;
 for(int i=lane;i<groups;i+=64)value+=partials[i*components+component];
 scratch[lane]=value;barrier();
 for(int step=32;step>0;step/=2){if(lane<step)scratch[lane]+=scratch[lane+step];barrier();}
 if(lane==0){if(isnan(scratch[0])||isinf(scratch[0]))atomicOr(bad,1u);results[resultOffset+component]=scratch[0];}
}
"""
