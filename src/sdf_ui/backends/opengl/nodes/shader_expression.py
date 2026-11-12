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
    token_pattern = re.compile(
        r"(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)|(?P<space>\s+)|"
        r"(?P<name>[A-Za-z_]\w*)|(?P<other>.)"
    )

    def tokens(text):
        return [
            (m.group(), m.start(), m.end())
            for m in token_pattern.finditer(text)
            if m.lastgroup not in {"comment", "space"}
        ]

    def replace_ranges(text, ranges):
        # Nested constructors in a removed tangent are already covered.
        selected = []
        for start, end in sorted(ranges):
            if selected and start < selected[-1][1]:
                if end > selected[-1][1]:
                    raise ValueError("Overlapping shader expression rewrites")
                continue
            selected.append((start, end))
        for start, end in reversed(selected):
            text = text[:start] + "0.0" + text[end:]
        return text

    items = tokens(source)
    constructors = []
    for i, (name, _, _) in enumerate(items):
        if name != "D" or i + 1 >= len(items) or items[i + 1][0] != "(":
            continue
        depth, comma = 1, None
        for token, start, end in items[i + 2 :]:
            if token == "(":
                depth += 1
            elif token == ")":
                depth -= 1
                if depth == 0:
                    if comma is None:
                        raise ValueError(
                            "Invalid dual constructor in shared shader program"
                        )
                    constructors.append((comma, start))
                    break
            elif token == "," and depth == 1:
                if comma is not None:
                    raise ValueError(
                        "Invalid dual constructor in shared shader program"
                    )
                comma = end
        else:
            raise ValueError("Invalid dual constructor in shared shader program")
    source = replace_ranges(source, constructors)

    items = tokens(source)
    tangent_reads = []
    i = 0
    while i < len(items):
        token, start, _ = items[i]
        if not re.fullmatch(r"[A-Za-z_]\w*", token) or (
            i > 0 and items[i - 1][0] == "."
        ):
            i += 1
            continue
        j = i + 1
        while j < len(items):
            if items[j][0] == "[":
                depth = 1
                j += 1
                while j < len(items) and depth:
                    depth += (items[j][0] == "[") - (items[j][0] == "]")
                    j += 1
                if depth:
                    raise ValueError("Unclosed shader array index")
            elif (
                items[j][0] == "."
                and j + 1 < len(items)
                and re.fullmatch(r"[A-Za-z_]\w*", items[j + 1][0])
            ):
                field = items[j + 1][0]
                j += 2
                if field == "d":
                    tangent_reads.append((start, items[j - 1][2]))
                    break
            else:
                break
        i = max(i + 1, j)
    return replace_ranges(source, tangent_reads)


@dataclass(frozen=True)
class Expression:
    op: str
    args: tuple = ()

    def emit(self, *, dual=False):
        if self.op == "literal":
            value = self.args[0]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Shader literals must be finite numbers")
            text = repr(float(value))
            return f"c({text})" if dual else text
        if self.op == "position":
            return "v2(vec2(pos))" if dual else "vec2(texelPos)"
        if self.op == "uniform":
            return ("U_" if dual else "") + self.args[0]
        if self.op == "load":
            return (
                f"ld({self.args[0]},pos,0)"
                if dual
                else f"imageLoad(source{self.args[0]},pos).r"
            )
        values = tuple(arg.emit(dual=dual) for arg in self.args)
        if self.op == "length2":
            return ("len2" if dual else "length") + f"({values[0]})"
        if self.op in {"sub", "sub2"}:
            if dual:
                return f"{self.op}({values[0]},{values[1]})"
            return f"({values[0]} - {values[1]})"
        if self.op in {"add", "mul", "div"}:
            if dual:
                return (
                    "divD" if self.op == "div" else self.op
                ) + f"({values[0]},{values[1]})"
            op = {"add": "+", "mul": "*", "div": "/"}[self.op]
            return f"({values[0]} {op} {values[1]})"
        if self.op in {"abs", "min", "max", "mix", "neg"}:
            name = (
                {
                    "abs": "absD",
                    "min": "minD",
                    "max": "maxD",
                    "mix": "mixD",
                    "neg": "neg",
                }[self.op]
                if dual
                else self.op
            )
            if not dual and self.op == "neg":
                return f"(-{values[0]})"
            return name + "(" + ",".join(values) + ")"
        raise ValueError(f"Unsupported shader expression operation '{self.op}'")

    def reverse_source(self, contract):
        """Emit a reverse-mode VJP for a scalar pointwise expression tree."""
        nodes = []
        offsets = {}
        offset = 0
        for arg in contract.parameters:
            offsets[arg.name] = (offset, arg.shape)
            offset += 1 if not arg.shape else arg.shape[0]

        def visit(node):
            children = tuple(
                visit(arg) for arg in node.args if isinstance(arg, Expression)
            )
            if node.op == "literal":
                kind, value = "float", node.emit()
            elif node.op == "position":
                kind, value = "vec2", "vec2(pos)"
            elif node.op == "load":
                if (
                    not 0 <= node.args[0] < len(contract.inputs)
                    or contract.inputs[node.args[0]].shape
                ):
                    raise ValueError("Reverse expression requires scalar image inputs")
                kind, value = "float", f"imageLoad(source{node.args[0]},pos).r"
            elif node.op == "uniform":
                start, shape = offsets[node.args[0]]
                kind = "float" if not shape else "vec2"
                value = (
                    f"parameterValues[uniformBase+{start}]"
                    if not shape
                    else f"vec2(parameterValues[uniformBase+{start}],parameterValues[uniformBase+{start + 1}])"
                )
                if shape not in ((), (2,)):
                    raise ValueError(
                        "Reverse expression supports scalar and vec2 uniforms"
                    )
            elif node.op in {"sub", "sub2"}:
                kind = "float" if node.op == "sub" else "vec2"
                value = f"(v{children[0]}-v{children[1]})"
            elif node.op in {"add", "mul", "div"}:
                op = {"add": "+", "mul": "*", "div": "/"}[node.op]
                kind, value = "float", f"(v{children[0]}{op}v{children[1]})"
            elif node.op == "length2":
                kind, value = "float", f"length(v{children[0]})"
            elif node.op in {"abs", "min", "max", "mix"}:
                kind, value = (
                    "float",
                    node.op + "(" + ",".join(f"v{x}" for x in children) + ")",
                )
            elif node.op == "neg":
                kind, value = "float", f"(-v{children[0]})"
            else:
                raise ValueError(f"Unsupported reverse expression '{node.op}'")
            index = len(nodes)
            nodes.append((node, children, kind, value))
            return index

        root = visit(self)
        if (
            nodes[root][2] != "float"
            or len(contract.outputs) != 1
            or contract.outputs[0].shape
        ):
            raise ValueError("Reverse expression requires a scalar output")
        forward = "".join(
            f"{kind} v{i}={value};" for i, (_, _, kind, value) in enumerate(nodes)
        )
        gradients = "".join(
            f"{kind} g{i}={kind}(0.0);" for i, (_, _, kind, _) in enumerate(nodes)
        )
        reverse = [f"g{root}=1.0;"]
        for i in range(len(nodes) - 1, -1, -1):
            node, children, _, _ = nodes[i]
            if node.op in {"sub", "sub2"}:
                reverse.append(f"g{children[0]}+=g{i};g{children[1]}-=g{i};")
            elif node.op == "add":
                reverse.append(f"g{children[0]}+=g{i};g{children[1]}+=g{i};")
            elif node.op == "mul":
                reverse.append(
                    f"g{children[0]}+=v{children[1]}*g{i};g{children[1]}+=v{children[0]}*g{i};"
                )
            elif node.op == "div":
                reverse.append(
                    f"g{children[0]}+=g{i}/v{children[1]};g{children[1]}-=g{i}*v{children[0]}/(v{children[1]}*v{children[1]});"
                )
            elif node.op == "length2":
                reverse.append(f"if(v{i}>0.0)g{children[0]}+=g{i}*v{children[0]}/v{i};")
            elif node.op == "neg":
                reverse.append(f"g{children[0]}-=g{i};")
            elif node.op == "abs":
                reverse.append(f"g{children[0]}+=g{i}*sign(v{children[0]});")
            elif node.op in {"min", "max"}:
                a, b = children
                sign = "<" if node.op == "min" else ">"
                reverse.append(
                    f"if(v{a}{sign}v{b})g{a}+=g{i};else if(v{b}{sign}v{a})g{b}+=g{i};else{{g{a}+=0.5*g{i};g{b}+=0.5*g{i};}}"
                )
            elif node.op == "mix":
                a, b, t = children
                reverse.append(
                    f"g{a}+=(1.0-v{t})*g{i};g{b}+=v{t}*g{i};g{t}+=(v{b}-v{a})*g{i};"
                )
        components = []
        for arg in contract.parameters:
            matching = [
                i
                for i, (node, _, _, _) in enumerate(nodes)
                if node.op == "uniform" and node.args[0] == arg.name
            ]
            if arg.shape == ():
                components.append("+".join(f"g{i}" for i in matching) or "0.0")
            elif arg.shape == (2,):
                components.extend(
                    ("+".join(f"g{i}.{axis}" for i in matching) or "0.0")
                    for axis in "xy"
                )
            else:
                raise ValueError("Reverse expression supports scalar and vec2 uniforms")
        values = "".join(
            f"gradient[{i}]=upstream*({value});" for i, value in enumerate(components)
        )
        count = len(components)
        declarations = "\n".join(
            f"layout(r32f,binding={i}) readonly uniform image2D source{i};"
            for i in range(len(contract.inputs))
        )
        scatter = []
        for index in range(len(contract.inputs)):
            matches = [
                i
                for i, (node, _, _, _) in enumerate(nodes)
                if node.op == "load" and node.args[0] == index
            ]
            if matches:
                total = "+".join(f"g{i}" for i in matches)
                scatter.append(
                    f"atomicAdjoint(inputOffsets[{index}]+pixel,upstream*({total}));"
                )
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

        declarations = ["#version 430", "layout(local_size_x=16,local_size_y=16) in;"]
        for index, arg in enumerate((*contract.outputs, *contract.inputs)):
            output = index < len(contract.outputs)
            fmt = sdf_format if arg.kind == "r32f" else color_format
            declarations.append(
                f"layout({fmt},binding={index}) {'writeonly' if output else 'readonly'} uniform image2D {arg.name};"
            )
        macros = []
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
                ""
                if match is None or match.group(1) is None
                else " = " + match.group(1)
            )
            array = arg.kind == "float" and len(arg.shape) == 1
            suffix = f"[{arg.shape[0]}]" if array else ""
            declarations.append(f"uniform {arg.kind} {arg.name}{suffix}{initializer};")
            if array:
                macros.append(f"#define U_{arg.name}(i) c({arg.name}[i])")
                continue
            if arg.kind == "float":
                expression = f"c({arg.name})"
            elif arg.kind in {"int", "bool"}:
                expression = f"c(float({arg.name}))"
            else:
                expression = f"v{arg.kind[-1]}({arg.name})"
            macros.append(f"#define U_{arg.name} {expression}")
        reads = []
        for index, arg in enumerate(contract.inputs):
            reads.append(
                f"if(which=={index}){{ivec2 n=imageSize({arg.name});"
                f"if(any(lessThan(p,ivec2(0)))||any(greaterThanEqual(p,n)))return c(0.0);"
                f"return c(imageLoad({arg.name},p)[channel]);}}"
            )
        stores = []
        offset = 0
        for arg in contract.outputs:
            channels = 1 if arg.kind == "r32f" else 4
            components = [f"o[{offset + i}].v" for i in range(channels)]
            offset += channels
            components.extend(["0.0"] * (4 - channels))
            stores.append(f"imageStore({arg.name},pos,vec4({','.join(components)}));")
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
