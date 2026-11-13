"""Small typed expression IR emitting GLSL and dual-number GLSL."""

import math
import re
from dataclasses import dataclass


def _primal_only(source):
    """Erase dual tangent expressions from a generated forward program.

    Forward execution never observes a tangent. Replacing each dual
    constructor's second argument and each tangent read allows the GLSL
    compiler to remove the remaining dead derivative-only branches.
    """
    items = _tokenize(source)
    source = _replace_ranges(source, _constructor_tangent_ranges(items))
    return _replace_ranges(source, _tangent_read_ranges(_tokenize(source)))


_TOKEN_PATTERN = re.compile(
    r"(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)|(?P<space>\s+)|"
    r"(?P<name>[A-Za-z_]\w*)|(?P<other>.)"
)
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def _tokenize(source):
    return [
        (match.group(), match.start(), match.end())
        for match in _TOKEN_PATTERN.finditer(source)
        if match.lastgroup not in {"comment", "space"}
    ]


def _replace_ranges(source, ranges):
    selected = []
    for start, end in sorted(ranges):
        if selected and start < selected[-1][1]:
            if end > selected[-1][1]:
                raise ValueError("Overlapping shader expression rewrites")
            continue
        selected.append((start, end))
    for start, end in reversed(selected):
        source = source[:start] + "0.0" + source[end:]
    return source


def _constructor_tangent_ranges(items):
    ranges = []
    for index, (name, _, _) in enumerate(items):
        if name == "D" and index + 1 < len(items) and items[index + 1][0] == "(":
            ranges.append(_constructor_tangent_range(items, index))
    return ranges


def _constructor_tangent_range(items, index):
    return _scan_constructor_arguments(items[index + 2 :])


def _scan_constructor_arguments(items):
    depth, comma = 1, None
    for token, start, end in items:
        match token:
            case "(":
                depth += 1
            case ")":
                depth -= 1
                if depth == 0:
                    return _finish_constructor_range(comma, start)
            case "," if depth == 1:
                comma = _record_constructor_comma(comma, end)
    raise ValueError("Invalid dual constructor in shared shader program")


def _record_constructor_comma(comma, end):
    if comma is not None:
        raise ValueError("Invalid dual constructor in shared shader program")
    return end


def _finish_constructor_range(comma, start):
    if comma is None:
        raise ValueError("Invalid dual constructor in shared shader program")
    return comma, start


def _tangent_read_ranges(items):
    ranges = []
    index = 0
    while index < len(items):
        item = _tangent_read_at(items, index)
        if item is None:
            index += 1
            continue
        index, value_range = item
        if value_range is not None:
            ranges.append(value_range)
    return ranges


def _tangent_read_at(items, index):
    token, start, _ = items[index]
    if not _IDENTIFIER.fullmatch(token) or (index > 0 and items[index - 1][0] == "."):
        return None
    cursor = index + 1
    while cursor < len(items):
        if items[cursor][0] == "[":
            cursor = _skip_array_index(items, cursor)
            continue
        if not _is_member_access(items, cursor):
            break
        field = items[cursor + 1][0]
        cursor += 2
        if field == "d":
            return cursor, (start, items[cursor - 1][2])
    return max(index + 1, cursor), None


def _skip_array_index(items, cursor):
    depth = 1
    cursor += 1
    while cursor < len(items) and depth:
        depth += (items[cursor][0] == "[") - (items[cursor][0] == "]")
        cursor += 1
    if depth:
        raise ValueError("Unclosed shader array index")
    return cursor


def _is_member_access(items, cursor):
    return (
        items[cursor][0] == "."
        and cursor + 1 < len(items)
        and _IDENTIFIER.fullmatch(items[cursor + 1][0])
    )


@dataclass(frozen=True)
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
            nodes[root][2] != "float"
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


def _emit_expression(op, args, dual):
    match op:
        case "literal":
            return _emit_literal(args[0], dual)
        case "position":
            return "v2(vec2(pos))" if dual else "vec2(texelPos)"
        case "uniform":
            return ("U_" if dual else "") + args[0]
        case "load":
            return (
                f"ld({args[0]},pos,0)" if dual else f"imageLoad(source{args[0]},pos).r"
            )
        case _:
            values = tuple(arg.emit(dual=dual) for arg in args)
            return _emit_operator(op, values, dual)


def _emit_literal(value, dual):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Shader literals must be finite numbers")
    text = repr(float(value))
    return f"c({text})" if dual else text


def _emit_operator(op, values, dual):
    if op == "length2":
        return ("len2" if dual else "length") + f"({values[0]})"
    if op in {"sub", "sub2"}:
        return (
            f"{op}({values[0]},{values[1]})" if dual else f"({values[0]} - {values[1]})"
        )
    if op in {"add", "mul", "div"}:
        return _emit_binary_operator(op, values, dual)
    if op in {"abs", "min", "max", "mix", "neg"}:
        return _emit_function_operator(op, values, dual)
    raise ValueError(f"Unsupported shader expression operation '{op}'")


def _emit_binary_operator(op, values, dual):
    if dual:
        return ("divD" if op == "div" else op) + f"({values[0]},{values[1]})"
    symbol = {"add": "+", "mul": "*", "div": "/"}[op]
    return f"({values[0]} {symbol} {values[1]})"


def _emit_function_operator(op, values, dual):
    if not dual and op == "neg":
        return f"(-{values[0]})"
    names = {"abs": "absD", "min": "minD", "max": "maxD", "mix": "mixD", "neg": "neg"}
    return (names[op] if dual else op) + "(" + ",".join(values) + ")"


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
            return "float", node.emit()
        case "position":
            return "vec2", "vec2(pos)"
        case "load":
            return _reverse_load_value(node, contract)
        case "uniform":
            return _reverse_uniform_value(node, offsets)
        case "sub" | "sub2":
            kind = "float" if node.op == "sub" else "vec2"
            return kind, f"(v{children[0]}-v{children[1]})"
        case "add" | "mul" | "div":
            symbol = {"add": "+", "mul": "*", "div": "/"}[node.op]
            return "float", f"(v{children[0]}{symbol}v{children[1]})"
        case "length2":
            return "float", f"length(v{children[0]})"
        case "abs" | "min" | "max" | "mix":
            return "float", node.op + "(" + ",".join(f"v{i}" for i in children) + ")"
        case "neg":
            return "float", f"(-v{children[0]})"
        case _:
            raise ValueError(f"Unsupported reverse expression '{node.op}'")


def _reverse_load_value(node, contract):
    index = node.args[0]
    if not 0 <= index < len(contract.inputs) or contract.inputs[index].shape:
        raise ValueError("Reverse expression requires scalar image inputs")
    return "float", f"imageLoad(source{index},pos).r"


def _reverse_uniform_value(node, offsets):
    start, shape = offsets[node.args[0]]
    if shape not in ((), (2,)):
        raise ValueError("Reverse expression supports scalar and vec2 uniforms")
    if not shape:
        return "float", f"parameterValues[uniformBase+{start}]"
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
        case "sub" | "sub2":
            return f"g{children[0]}+=g{index};g{children[1]}-=g{index};"
        case "add":
            return f"g{children[0]}+=g{index};g{children[1]}+=g{index};"
        case "mul":
            return f"g{children[0]}+=v{children[1]}*g{index};g{children[1]}+=v{children[0]}*g{index};"
        case "div":
            return f"g{children[0]}+=g{index}/v{children[1]};g{children[1]}-=g{index}*v{children[0]}/(v{children[1]}*v{children[1]});"
        case "length2":
            return f"if(v{index}>0.0)g{children[0]}+=g{index}*v{children[0]}/v{index};"
        case "neg":
            return f"g{children[0]}-=g{index};"
        case "abs":
            return f"g{children[0]}+=g{index}*sign(v{children[0]});"
        case "min" | "max":
            return _reverse_minmax_rule(op, children, index)
        case "mix":
            a, b, weight = children
            return f"g{a}+=(1.0-v{weight})*g{index};g{b}+=v{weight}*g{index};g{weight}+=(v{b}-v{a})*g{index};"
        case _:
            return ""


def _reverse_minmax_rule(op, children, index):
    left, right = children
    relation = "<" if op == "min" else ">"
    return (
        f"if(v{left}{relation}v{right})g{left}+=g{index};"
        f"else if(v{right}{relation}v{left})g{right}+=g{index};"
        f"else{{g{left}+=0.5*g{index};g{right}+=0.5*g{index};}}"
    )


def _reverse_uniform_components(contract, nodes):
    components = []
    for argument in contract.parameters:
        matches = _matching_uniform_nodes(nodes, argument.name)
        if argument.shape == ():
            components.append("+".join(f"g{i}" for i in matches) or "0.0")
        elif argument.shape == (2,):
            components.extend(
                "+".join(f"g{i}.{axis}" for i in matches) or "0.0" for axis in "xy"
            )
        else:
            raise ValueError("Reverse expression supports scalar and vec2 uniforms")
    return components


def _matching_uniform_nodes(nodes, name):
    return [
        index
        for index, (node, _, _, _) in enumerate(nodes)
        if node.op == "uniform" and node.args[0] == name
    ]


def _reverse_input_declarations(contract):
    return "\n".join(
        f"layout(r32f,binding={index}) readonly uniform image2D source{index};"
        for index in range(len(contract.inputs))
    )


def _reverse_input_scatter(contract, nodes):
    scatter = []
    for index in range(len(contract.inputs)):
        matches = [
            node_index
            for node_index, (node, _, _, _) in enumerate(nodes)
            if node.op == "load" and node.args[0] == index
        ]
        if matches:
            total = "+".join(f"g{node_index}" for node_index in matches)
            scatter.append(
                f"atomicAdjoint(inputOffsets[{index}]+pixel,upstream*({total}));"
            )
    return scatter


@dataclass(frozen=True)
class ShaderProgram:
    """A shared typed shader program compiled to primal and local VJP GLSL.

    The program uses the dual arithmetic language, which also represents
    branch selection and sampled image reads. A zero tangent gives the primal
    evaluation; seeded tangents provide the local backward evaluation.
    """

    name: str
    body: str
    reverse_expression: object = None

    def forward(
        self, contract, *, sdf_format="r32f", color_format="rgba8", original_source=""
    ):
        from ..dual_glsl import PRELUDE

        declarations, macros = _shader_declarations(
            contract, sdf_format, color_format, original_source
        )
        reads = _shader_input_reads(contract)
        stores = _shader_output_stores(contract)
        source = "\n".join(
            declarations
            + ["ivec2 pixelSize;", PRELUDE]
            + macros
            + [
                "D ld(int which,ivec2 p,int channel){"
                + "".join(reads)
                + "return c(0.0);}",
                re.sub(r"\bsize\b", "pixelSize", self.body),
                "void main(){ivec2 pos=ivec2(gl_GlobalInvocationID.xy);"
                + f"pixelSize=imageSize({contract.outputs[0].name});"
                + "if(any(greaterThanEqual(pos,pixelSize)))return;D o[8];evaluate(pos,o);"
                + "".join(stores)
                + "}",
            ]
        )
        # A shader may have a uniform named `c`, as Bezier does. Keep public
        # uniform names while avoiding the dual constructor's identifier.
        return _primal_only(re.sub(r"\bc\s*\(", "dualC(", source))


def _shader_declarations(contract, sdf_format, color_format, original_source):
    declarations = ["#version 430", "layout(local_size_x=16,local_size_y=16) in;"]
    for index, arg in enumerate((*contract.outputs, *contract.inputs)):
        output = index < len(contract.outputs)
        fmt = sdf_format if arg.kind == "r32f" else color_format
        access = "writeonly" if output else "readonly"
        declarations.append(
            f"layout({fmt},binding={index}) {access} uniform image2D {arg.name};"
        )
    parameter_declarations, macros = _shader_parameter_declarations(
        contract, original_source
    )
    return declarations + parameter_declarations, macros


def _shader_parameter_declarations(contract, original_source):
    declarations, macros = [], []
    for arg in contract.parameters:
        match = re.search(
            r"\buniform\s+"
            + re.escape(arg.kind)
            + r"\s+"
            + re.escape(arg.name)
            + r"(?:\s*\[\s*\w+\s*\])?\s*(?:=\s*([^;]+))?\s*;",
            original_source,
            re.DOTALL,
        )
        initializer = (
            "" if match is None or match.group(1) is None else " = " + match.group(1)
        )
        array = arg.kind == "float" and len(arg.shape) == 1
        suffix = f"[{arg.shape[0]}]" if array else ""
        declarations.append(f"uniform {arg.kind} {arg.name}{suffix}{initializer};")
        macros.append(_shader_parameter_macro(arg, array))
    return declarations, macros


def _shader_parameter_macro(arg, array):
    if array:
        return f"#define U_{arg.name}(i) c({arg.name}[i])"
    if arg.kind == "float":
        expression = f"c({arg.name})"
    elif arg.kind in {"int", "bool"}:
        expression = f"c(float({arg.name}))"
    else:
        expression = f"v{arg.kind[-1]}({arg.name})"
    return f"#define U_{arg.name} {expression}"


def _shader_input_reads(contract):
    return [
        f"if(which=={index}){{ivec2 n=imageSize({arg.name});"
        f"if(any(lessThan(p,ivec2(0)))||any(greaterThanEqual(p,n)))return c(0.0);"
        f"return c(imageLoad({arg.name},p)[channel]);}}"
        for index, arg in enumerate(contract.inputs)
    ]


def _shader_output_stores(contract):
    stores, offset = [], 0
    for arg in contract.outputs:
        channels = 1 if arg.kind == "r32f" else 4
        components = [f"o[{offset + index}].v" for index in range(channels)]
        offset += channels
        components.extend(["0.0"] * (4 - channels))
        stores.append(f"imageStore({arg.name},pos,vec4({','.join(components)}));")
    return stores
