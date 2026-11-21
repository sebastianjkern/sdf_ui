"""Small typed expression IR emitting GLSL and dual-number GLSL."""




from ._shader_expression_emit import _emit_expression
from ._shader_expression_inputs import (
    _reverse_input_declarations,
    _reverse_input_scatter,
    _reverse_uniform_components,
)
from ._shader_expression_primal import _primal_only  # noqa: F401

_FLOAT_TEXT = 'float'

class Expression:
    op: str
    args: tuple = ()

    def emit(self, *, dual=False):
        return _emit_expression(self.op, self.args, dual)

    def reverse_source(self, contract):
        """Emit a reverse-mode VJP for a scalar pointwise expression tree."""
        nodes = []
        offsets = _reverse_uniform_offsets(contract)
        root = _visit_reverse_expression(self, contract, offsets, nodes)
        if (
            nodes[root][2] != _FLOAT_TEXT
            or len(contract.outputs) != 1
            or contract.outputs[0].shape
        ):
            raise ValueError("Reverse expression requires a scalar output")
        forward = _reverse_forward(nodes)
        gradients = _reverse_gradient_variables(nodes)
        reverse = _reverse_rules(nodes, root)
        components = _reverse_uniform_components(contract, nodes)
        values = "".join(
            f"gradient[{i}]=upstream*({value});" for i, value in enumerate(components)
        )
        count = len(components)
        declarations = _reverse_input_declarations(contract)
        scatter = _reverse_input_scatter(contract, nodes)
        return f"""#version 430
layout(local_size_x=64) in;
{declarations}
layout(std430,binding=0) readonly buffer Parameters {{ float parameterValues[]; }};
layout(std430,binding=1) buffer AdjointValues {{ uint adjoints[]; }};
layout(std430,binding=2) writeonly buffer Partials {{ float partials[]; }};
layout(std430,binding=4) buffer Status {{ uint bad; }};
uniform ivec2 size;
uniform int outputOffsets[1];
uniform int inputOffsets[{max(len(contract.inputs), 1)}];
uniform int uniformBase;
shared float scratch[64];
void atomicAdjoint(int index,float value){{
 if(value==0.0)return;
 if(isnan(value)||isinf(value)){{atomicOr(bad,1u);return;}}
 uint old=atomicCompSwap(adjoints[index],0u,0u),expected;
 do{{expected=old;float next=uintBitsToFloat(expected)+value;
 if(isnan(next)||isinf(next)){{atomicOr(bad,1u);return;}}
 old=atomicCompSwap(adjoints[index],expected,floatBitsToUint(next));}}while(old!=expected);
}}
void main(){{
 int pixel=int(gl_GlobalInvocationID.x),lane=int(gl_LocalInvocationID.x);
 float gradient[{max(count, 1)}];
 for(int i=0;i<{count};i++)gradient[i]=0.0;
 if(pixel<size.x*size.y){{
  ivec2 pos=ivec2(pixel%size.x,pixel/size.x);
  float upstream=uintBitsToFloat(adjoints[outputOffsets[0]+pixel]);
  if(upstream!=0.0){{{forward}{gradients}{"".join(reverse)}{values}{"".join(scatter)}}}
 }}
 for(int i=0;i<{count};i++){{
  float value=gradient[i];
  if(isnan(value)||isinf(value)){{atomicOr(bad,1u);value=0.0;}}
  scratch[lane]=value;barrier();
  for(int stride=32;stride>0;stride/=2){{if(lane<stride)scratch[lane]+=scratch[lane+stride];barrier();}}
  if(lane==0)partials[int(gl_WorkGroupID.x)*{max(count, 1)}+i]=scratch[0];
  barrier();
 }}
}}
"""


def _reverse_uniform_offsets(contract):
    offsets = {}
    offset = 0
    for argument in contract.parameters:
        offsets[argument.name] = offset, argument.shape
        offset += 1 if not argument.shape else argument.shape[0]
    return offsets


def _visit_reverse_expression(node, contract, offsets, nodes):
    children = tuple(
        _visit_reverse_expression(arg, contract, offsets, nodes)
        for arg in node.args
        if isinstance(arg, Expression)
    )
    kind, value = _reverse_node_value(node, children, contract, offsets)
    nodes.append((node, children, kind, value))
    return len(nodes) - 1


def _reverse_node_value(node, children, contract, offsets):
    match node.op:
        case "literal":
            return _FLOAT_TEXT, node.emit()
        case "position":
            return "vec2", "vec2(pos)"
        case "load":
            return _reverse_load_value(node, contract)
        case "uniform":
            return _reverse_uniform_value(node, offsets)
        case _:
            return _reverse_arithmetic_value(node, children)


def _reverse_arithmetic_value(node, children):
    if node.op in {"sub", "sub2"}:
        kind = _FLOAT_TEXT if node.op == "sub" else "vec2"
        return kind, f"(v{children[0]}-v{children[1]})"
    if node.op in {"add", "mul", "div"}:
        symbol = {"add": "+", "mul": "*", "div": "/"}[node.op]
        return _FLOAT_TEXT, f"(v{children[0]}{symbol}v{children[1]})"
    if node.op == "length2":
        return _FLOAT_TEXT, f"length(v{children[0]})"
    if node.op in {"abs", "min", "max", "mix"}:
        return _FLOAT_TEXT, node.op + "(" + ",".join(f"v{i}" for i in children) + ")"
    if node.op == "neg":
        return _FLOAT_TEXT, f"(-v{children[0]})"
    raise ValueError(f"Unsupported reverse expression '{node.op}'")


def _reverse_load_value(node, contract):
    index = node.args[0]
    if not 0 <= index < len(contract.inputs) or contract.inputs[index].shape:
        raise ValueError("Reverse expression requires scalar image inputs")
    return _FLOAT_TEXT, f"imageLoad(source{index},pos).r"


def _reverse_uniform_value(node, offsets):
    start, shape = offsets[node.args[0]]
    if shape not in ((), (2,)):
        raise ValueError("Reverse expression supports scalar and vec2 uniforms")
    if not shape:
        return _FLOAT_TEXT, f"parameterValues[uniformBase+{start}]"
    value = f"vec2(parameterValues[uniformBase+{start}],parameterValues[uniformBase+{start + 1}])"
    return "vec2", value


def _reverse_forward(nodes):
    return "".join(
        f"{kind} v{index}={value};" for index, (_, _, kind, value) in enumerate(nodes)
    )


def _reverse_gradient_variables(nodes):
    return "".join(
        f"{kind} g{index}={kind}(0.0);" for index, (_, _, kind, _) in enumerate(nodes)
    )


def _reverse_rules(nodes, root):
    rules = [f"g{root}=1.0;"]
    for index in range(len(nodes) - 1, -1, -1):
        node, children, _, _ = nodes[index]
        rule = _reverse_rule(node.op, children, index)
        if rule:
            rules.append(rule)
    return "".join(rules)


def _reverse_rule(op, children, index):
    match op:
        case "min" | "max":
            return _reverse_minmax_rule(op, children, index)
        case "mix":
            a, b, weight = children
            return f"g{a}+=(1.0-v{weight})*g{index};g{b}+=v{weight}*g{index};g{weight}+=(v{b}-v{a})*g{index};"
        case _:
            return _reverse_arithmetic_rule(op, children, index)


def _reverse_arithmetic_rule(op, children, index):
    if op in {"sub", "sub2"}:
        return f"g{children[0]}+=g{index};g{children[1]}-=g{index};"
    if op == "add":
        return f"g{children[0]}+=g{index};g{children[1]}+=g{index};"
    if op == "mul":
        return f"g{children[0]}+=v{children[1]}*g{index};g{children[1]}+=v{children[0]}*g{index};"
    if op == "div":
        return f"g{children[0]}+=g{index}/v{children[1]};g{children[1]}-=g{index}*v{children[0]}/(v{children[1]}*v{children[1]});"
    if op == "length2":
        return f"if(v{index}>0.0)g{children[0]}+=g{index}*v{children[0]}/v{index};"
    if op == "neg":
        return f"g{children[0]}-=g{index};"
    if op == "abs":
        return f"g{children[0]}+=g{index}*sign(v{children[0]});"
    return ""


def _reverse_minmax_rule(op, children, index):
    left, right = children
    relation = "<" if op == "min" else ">"
    return (
        f"if(v{left}{relation}v{right})g{left}+=g{index};"
        f"else if(v{right}{relation}v{left})g{right}+=g{index};"
        f"else{{g{left}+=0.5*g{index};g{right}+=0.5*g{index};}}"
    )












