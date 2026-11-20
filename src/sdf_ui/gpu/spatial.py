"""Compile coordinate-dependent parameter bindings into one reusable WGSL pass."""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from ..core.expressions import Expr
from ..core.spatial import is_spatial, spatial_scene
from ..core.texture import _SPATIAL_PARAMS
from .nodes.common import build
from .nodes.registry import registry
from .regions import _matching_brace, _matching_paren, _split_arguments

_TRANSFORMS = {"translate", "rotate", "scale", "skew"}
_INLINE_COMPOSITES = {
    "radial_gradient",
    "linear_gradient",
    "ngon",
    "polygon",
    "convex_polygon",
}
_VECTOR = {
    f"{prefix}vec{size}": size for prefix in ("", "i", "u") for size in (2, 3, 4)
}
_ARRAY = re.compile(r"(float|int|uint)\[(\d+)\]")
_WGSL_TYPES = {"float": "f32", "int": "i32", "uint": "u32"}
_BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_FUNCTIONS = {
    "sin": "sin",
    "cos": "cos",
    "sqrt": "sqrt",
    "abs": "abs",
    "tanh": "tanh",
    "maximum": "max",
    "minimum": "min",
    "pow": "pow",
}
_IMAGE_BINDING = re.compile(
    r"@group\(0\)\s*@binding\(\d+\)\s*var\s+(\w+)\s*:\s*texture_storage_2d<[^>]+>\s*;"
)
_UNIFORM_BINDING = re.compile(
    r"@group\(0\)\s*@binding\(\d+\)\s*var<uniform>\s+(\w+)\s*:\s*(\w+)\s*;"
)
_ENTRY = re.compile(
    r"@compute\s*@workgroup_size\([^)]*\)\s*fn\s+compute_main\(\s*"
    r"@builtin\(global_invocation_id\)\s*(\w+)\s*:[^{]+\{"
)


@dataclass(frozen=True)
class Kernel:
    descriptor: object
    source: str
    output: str


def _replace_calls(code, function, replace):
    """Replace WGSL calls with balanced argument lists, including inner calls."""
    pattern = re.compile(r"\b" + function + r"\(")
    offset = 0
    while (match := pattern.search(code, offset)) is not None:
        opening = match.end() - 1
        end = _matching_paren(code, opening)
        args = _split_arguments(code[opening + 1 : end])
        replacement = replace(*(arg.strip() for arg in args))
        code = code[: match.start()] + replacement + code[end + 1 :]
        offset = match.start() + len(replacement)
    return code


def _make_kernel(shader):
    if shader.wgsl is not None:
        return shader.wgsl.forward()
    root = Path(__file__).with_name("shaders")
    variant = root / f"{shader.name}_rgba32f.wgsl"
    path = variant if variant.is_file() else root / f"{shader.name}.wgsl"
    if not path.is_file():
        raise NotImplementedError(
            f"Spatial shader '{shader.name}' has no native source"
        )
    return path.read_text(encoding="utf-8")


def _color_output(node, tex, context):
    from ..core.color import ColorTexture

    return ColorTexture(tex=tex, context=context, mode=node.mode or "LAB")


class Compiler:
    """A compilation has no image readbacks and retains no materialized images."""

    def __init__(self, renderer=None):
        self.renderer = renderer
        self.ctx = renderer.ctx if renderer is not None else None
        self.scale = getattr(self.ctx, "device_scale", 1.0)
        self.functions = []
        self.constants = []
        self.maps = []
        self.map_indices = {}
        self.memo = {}
        self.nodes = []

    def constant(self, value):
        if self.renderer is not None and (
            not isinstance(value, Real) or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"Expected a finite numeric spatial uniform, got {value!r}"
            )
        index = len(self.constants)
        self.constants.append(float(value) if self.renderer is not None else value)
        return f"spatial_value({index})"

    def resolve(self, value):
        if self.renderer is not None:
            return self.renderer._resolve(value)
        if isinstance(value, str) and value.startswith("#"):
            from ..util import hex_col

            return hex_col(value)
        return value

    def texture(self, node):
        """Use an existing graph output as a read-only texture binding."""
        if id(node) not in self.map_indices:
            result = None if self.renderer is None else self.renderer.render(node)
            if result is not None and result.tex.context is not self.ctx:
                raise ValueError("Spatial textures must belong to the rendering canvas")
            self.map_indices[id(node)] = len(self.maps)
            self.maps.append((node, result))
        return self.map_indices[id(node)]

    def expression(self, value):
        if not is_spatial(value):
            return self.constant(self.resolve(value))
        if not isinstance(value, Expr):
            raise ValueError(
                "Use a tuple of spatial expressions for a vector parameter"
            )
        op = value.op
        if op == "texture_sample":
            texture, index, scope = value.args
            position = "world" if scope == "pixel" else "cell"
            return f"sample_{self.texture(texture)}({position})[{index}]"
        if op == "index":
            raw, index = value.args
            if isinstance(raw, (tuple, list)) and isinstance(index, int):
                return self.expression(raw[index])
            raise ValueError("Index a spatial vector with a fixed component index")
        args = [self.expression(arg) for arg in value.args]
        if op in _BINARY:
            return f"({args[0]} {_BINARY[op]} {args[1]})"
        if op in _FUNCTIONS:
            return f"{_FUNCTIONS[op]}({', '.join(args)})"
        if op == "neg":
            return f"(-{args[0]})"
        if op == "relu":
            return f"max(0.0, {args[0]})"
        if op == "softplus":
            return f"(max({args[0]},0.0)+log(1.0+exp(-abs({args[0]}))))"
        if op == "sigmoid":
            return f"(1.0/(1.0+exp(-{args[0]})))"
        if op == "select_positive":
            return f"select({args[2]}, {args[1]}, ({args[0]}>0.0))"
        if op.startswith("percent"):
            if self.ctx is None:
                return f"({args[0]} * {self.constant(Expr(op, (1.0,)))})"
            dimension = {
                "percent": self.ctx.logical_size[0],
                "percent_x": self.ctx.logical_size[0],
                "percent_y": self.ctx.logical_size[1],
                "percent_of_min": min(self.ctx.logical_size),
            }.get(op)
            if dimension is not None:
                return f"({args[0]} * {self.constant(dimension / 100)})"
        raise NotImplementedError(f"Expression '{op}' cannot be sampled on the GPU")

    def uniform(self, value, glsl_type, *, scale=1):
        def scalar(part):
            result = self.expression(part)
            return result if scale == 1 else f"({result} * {self.constant(scale)})"

        array = _ARRAY.fullmatch(glsl_type)
        if array is not None:
            item_type, count = array.groups()
            items = value if isinstance(value, (list, tuple)) else self.resolve(value)
            if self.renderer is None and not hasattr(items, "__len__"):
                items = tuple(Expr("index", (items, i)) for i in range(int(count)))
            if not hasattr(items, "__len__") or len(items) != int(count):
                raise ValueError(f"Spatial {glsl_type} requires {count} components")
            return (
                f"array<{_WGSL_TYPES[item_type]},{count}>("
                + ",".join(self.uniform(item, item_type, scale=scale) for item in items)
                + ")"
            )
        if glsl_type in _VECTOR:
            parts = value if isinstance(value, (tuple, list)) else self.resolve(value)
            size = _VECTOR[glsl_type]
            if (
                glsl_type == "vec4"
                and isinstance(parts, (tuple, list))
                and len(parts) == 3
            ):
                parts = (*parts, 1.0)
            if self.renderer is None and not hasattr(parts, "__len__"):
                parts = tuple(Expr("index", (parts, i)) for i in range(size))
            if not hasattr(parts, "__len__") or len(parts) != size:
                raise ValueError(f"Spatial {glsl_type} requires {size} components")
            component = {"i": "i32", "u": "u32"}.get(glsl_type[0], "f32")
            expressions = [scalar(part) for part in parts]
            if component != "f32":
                expressions = [f"{component}({part})" for part in expressions]
            return f"vec{size}<{component}>({', '.join(expressions)})"
        result = scalar(value)
        if glsl_type == "float":
            return result
        if glsl_type in {"int", "uint"}:
            return f"{_WGSL_TYPES[glsl_type]}({result})"
        if glsl_type == "bool":
            return f"({result} != 0.0)"
        raise NotImplementedError(f"Spatial binding for uniform type '{glsl_type}'")

    def sampler(self, i, node, rendered):
        image = rendered.tex
        # The currently allocated rectangle is independent of canvas coordinates.
        x, y = map(self.constant, image.origin)
        width, height = map(self.constant, image.canvas_size)
        fmt = image.format
        return f"""
@group(0) @binding({i + 2}) var tex_{i}: texture_storage_2d<{fmt}, read>;
fn sample_{i}(p: vec2<f32>) -> vec4<f32> {{
    let extent = vec2<f32>({width}, {height});
    let q = clamp(p, vec2<f32>(0.0), extent-1.0);
    let lo = vec2<i32>(floor(q));
    let hi = min(lo+1, vec2<i32>(extent)-1);
    let t = fract(q);
    let origin = vec2<i32>(i32({x}), i32({y}));
    return mix(
        mix(textureLoad(tex_{i}, lo-origin), textureLoad(tex_{i}, vec2<i32>(hi.x,lo.y)-origin), t.x),
        mix(textureLoad(tex_{i}, vec2<i32>(lo.x,hi.y)-origin), textureLoad(tex_{i}, hi-origin), t.x), t.y);
}}
"""

    def evaluate(self, node, *, baked=False, output=0):
        key = (id(node), baked, output)
        if key in self.memo:
            return self.memo[key]
        name = f"op_{len(self.memo)}"
        self.memo[key] = name
        self.nodes.append(node)
        if getattr(node, "tex", None) is not None or baked:
            body = f"return sample_{self.texture(node)}(p);"
        elif node.op == "output":
            index = dict(node.params)["index"]
            body = f"return {self.evaluate(node.inputs[0], output=index)}(p);"
        elif node.op == "spatial_repeat":
            body = self.repeat(node)
        elif node.op in _TRANSFORMS:
            body = self.transform(node)
        else:
            body = self.shader(node, name, output)
            if body is None:
                return name
        self.functions.append(f"fn {name}(p: vec2<f32>) -> vec4<f32> {{\n{body}\n}}")
        return name

    def repeat(self, node):
        raw = dict(node.params)
        child = node.inputs[0]
        # Finished textures can be stamped with mapped transforms as well.
        source = self.evaluate(
            child, baked=child.tex is not None or raw.get("baked", False)
        )
        spacing = self.uniform(raw["spacing"], "vec2", scale=self.scale)
        center = self.uniform(raw["center"], "vec2", scale=self.scale)
        origin = self.uniform(raw["stamp_center"], "vec2", scale=self.scale)
        if raw["columns"] is None:
            grid_offset = "vec2<f32>(0.0)"
            clipping = ""
        else:
            counts = self.uniform((raw["columns"], raw["rows"]), "vec2")
            grid_offset = f"({counts}-1.0)*0.5"
            clipping = (
                f"if any(index < vec2<f32>(0.0)) || any(index >= {counts}) "
                "{ return outside; }"
            )
        rotation = self.uniform(raw["rotation"], "float")
        scale = self.uniform(raw["scale"], "float")
        outside = (
            "vec4<f32>(max(spacing.x, spacing.y),0,0,0)"
            if node.kind == "sdf"
            else "vec4<f32>(0.0)"
        )
        distance_scale = "out.x *= abs(factor);" if node.kind == "sdf" else ""
        return f"""
    let spacing = {spacing};
    if any(spacing <= vec2<f32>(0.0)) {{ return {outside}; }}
    let outside = {outside};
    let offset = {grid_offset};
    let center = {center};
    let index = floor((p-center)/spacing + offset + 0.5);
    {clipping}
    let cell_local = center + (index-offset)*spacing;
    let last_cell = cell;
    let last_basis = basis;
    cell = world + basis * (cell_local - p);
    let angle = {rotation};
    let factor = {scale};
    if abs(factor) < 0.000001 {{ cell=last_cell; return {outside}; }}
    let delta = p-cell_local;
    let co=cos(angle); let si=sin(angle);
    let uv = vec2<f32>(co*delta.x+si*delta.y, -si*delta.x+co*delta.y)/factor + {origin};
    basis = basis * mat2x2<f32>(vec2<f32>(co, si)*factor, vec2<f32>(-si, co)*factor);
    var out = {source}(uv);
    {distance_scale}
    cell = last_cell;
    basis = last_basis;
    return out;
"""

    def transform(self, node):
        raw = dict(node.params)
        source = self.evaluate(node.inputs[0])
        if node.op == "translate":
            offset = self.uniform(raw["offset"], "vec2", scale=self.scale)
            return f"return {source}(p - {offset});"
        center = self.uniform(raw["center"], "vec2", scale=self.scale)
        if node.op == "rotate":
            angle = self.uniform(raw["angle"], "float")
            return f"""
    let d=p-{center}; let angle={angle}; let co=cos(angle); let si=sin(angle);
    let previous = basis;
    basis = basis * mat2x2<f32>(vec2<f32>(co,si), vec2<f32>(-si,co));
    let result = {source}(vec2<f32>(co*d.x+si*d.y,-si*d.x+co*d.y)+{center});
    basis = previous;
    return result;
"""
        if node.op == "scale":
            factor = self.uniform(raw["factor"], "float")
            return f"""
    let d=p-{center}; let factor={factor};
    if abs(factor)<0.000001 {{ return vec4<f32>(1.0e6,0,0,0); }}
    let previous = basis;
    basis = basis * factor;
    var out={source}(d/factor+{center});
    basis = previous;
    out.x*=abs(factor);
    return out;
"""
        skew = self.uniform(raw["skew"], "vec2")
        return f"""
    let d=p-{center}; let skew={skew}; let det=1.0-skew.x*skew.y;
    if abs(det)<0.000001 {{ return vec4<f32>(1.0e6,0,0,0); }}
    let inverse = vec2<f32>(d.x-skew.x*d.y,d.y-skew.y*d.x)/det;
    let previous = basis;
    basis = basis * mat2x2<f32>(vec2<f32>(1,skew.y),vec2<f32>(skew.x,1));
    let result = {source}(inverse+{center});
    basis = previous;
    return result;
"""

    def shader(self, node, name, output):
        plugin = registry.get(node.op)
        raw = dict(node.params)
        if plugin.spatial_lowering is not None:
            lowered = plugin.spatial_lowering(node.inputs, raw)
            return f"return {self.evaluate(lowered, output=output)}(p);"
        # Reuse ordinary plugin compositions where no custom raster pass is needed.
        if node.op in {"ngon", "polygon", "convex_polygon"}:
            return self.polygon(node, raw)
        if node.op in {"radial_gradient", "linear_gradient"}:
            from ..core.expressions import _lazy_scalar

            if node.op == "radial_gradient":
                shape = build("circle", raw["a"], 0)
                inner, outer = raw["inner"], raw["outer"]
            else:
                ax, ay = map(_lazy_scalar, raw["a"])
                bx, by = map(_lazy_scalar, raw["b"])
                dx, dy = ax - bx, ay - by
                shape = build("line", (ax + dx, ay - dy), (ax - dx, ay + dy))
                inner, outer = 0, Expr("sqrt", (dx * dx + dy * dy,))
            child = build(
                "fill",
                shape,
                raw["color1"],
                raw["color2"],
                0,
                inner,
                outer,
            )
            self.functions.append(
                f"fn {name}(p: vec2<f32>) -> vec4<f32> {{ return {self.evaluate(child)}(p); }}"
            )
            return None
        if plugin.shader is None or not plugin.shader.interface:
            if not spatial_scene(node):
                return f"return sample_{self.texture(node)}(p);"
            raise NotImplementedError(
                f"'{node.op}' cannot consume spatial parameters because it has no callable shader"
            )
        shader = plugin.shader
        interface = shader.interface
        bindings = (interface.output, *interface.extra_outputs)
        if not 0 <= output < len(bindings):
            raise ValueError(f"Shader '{shader.name}' has no output {output}")
        params = self.uniform_values(node, plugin, raw)
        if node.op in {"alpha_overlay", "mask", "layer_mask", "multiply"}:
            children = tuple(child.to_rgb() for child in node.inputs)
        else:
            children = node.inputs
        input_names = plugin.input_uniforms or tuple(
            item.name for item in interface.inputs
        )
        child_functions = dict(
            zip(input_names, (self.evaluate(child) for child in children))
        )
        code = _make_kernel(shader)
        self.functions.append(
            self.inline(
                Kernel(shader, code, bindings[output].name),
                name,
                params,
                child_functions,
            )
        )
        return None

    def polygon(self, node, raw):
        """A shared coordinate evaluator for polygons with any fixed topology."""
        from ..core.expressions import _lazy_scalar, cos, sin

        if node.op == "ngon":
            count = raw["sides"]
            if not isinstance(count, int) or count < 3:
                raise ValueError(
                    "Spatial polygons require a fixed side count of at least three"
                )
            cx, cy = map(_lazy_scalar, raw["center"])
            radius = _lazy_scalar(raw["radius"])
            rotation = _lazy_scalar(raw["rotation"])
            angles = (rotation + 2 * math.pi * i / count for i in range(count))
            points = tuple((cx + cos(a) * radius, cy + sin(a) * radius) for a in angles)
        else:
            points = raw["points"]
            if not isinstance(points, (tuple, list)) or len(points) < 3:
                raise ValueError(
                    "Spatial polygons require a fixed list of at least three points"
                )
            count = len(points)
        vertices = ", ".join(
            self.uniform(point, "vec2", scale=self.scale) for point in points
        )
        return f"""
    let vertices = array<vec2<f32>,{count}>({vertices});
    var distance2 = 1.0e20;
    var inside = false;
    for (var i=0; i<{count}; i++) {{
        let a=vertices[i]; let b=vertices[(i+1)%{count}];
        let edge=b-a; let delta=p-a;
        let u=clamp(dot(delta,edge)/max(dot(edge,edge),1.0e-12),0.0,1.0);
        let separation=delta-u*edge;
        distance2=min(distance2,dot(separation,separation));
        if (a.y>p.y)!=(b.y>p.y) {{
            if p.x < a.x + edge.x*(p.y-a.y)/edge.y {{ inside = !inside; }}
        }}
    }}
    return vec4<f32>(select(sqrt(distance2),-sqrt(distance2),inside),0,0,0);
"""

    def uniform_values(self, node, plugin, raw):
        """Apply existing plugin aliases while leaving sampled expressions intact."""
        interface = plugin.shader.interface
        names = {item.name for item in interface.uniforms}
        if not names:
            return {}
        shader_name = plugin.shader.name
        adapter = plugin.spatial_uniforms or plugin.make_uniforms
        if adapter is not None:
            try:
                values = adapter(raw)
            except (ValueError, TypeError) as exc:
                if (
                    plugin.spatial_uniforms is None
                    and names.issubset(raw)
                    and (
                        shader_name.startswith("hdr_")
                        or node.kind == "field"
                        or node.op in {"warp_sdf", "warp_color"}
                    )
                ):
                    # These builtins validate uniform shape and range at the
                    # render boundary. Their symbolic mapping is identity.
                    values = raw
                else:
                    raise ValueError(
                        f"Spatial uniforms for '{node.op}' need a symbolic adapter"
                    ) from exc
        else:
            values = raw
        result = {}
        for item in interface.uniforms:
            value = values.get(item.name, 0)
            if item.name == "corner_radius" and not isinstance(value, (tuple, list)):
                value = (value,) * 4
            # Shader aliases such as circle.offset and fill.first are measures
            # even when the public parameter names differ.
            spatial = item.name in _SPATIAL_PARAMS or item.name in {
                "first",
                "second",
                "smoothness",
                "repeat",
            }
            result[item.name] = self.uniform(
                value, item.glsl_type, scale=self.scale if spatial else 1
            )
        return result

    def inline(self, kernel, name, params, children):
        """Adapt the existing shader's entry point to a coordinate function."""
        code = kernel.source
        match = _ENTRY.search(code)
        if match is None:
            raise NotImplementedError(
                f"'{kernel.descriptor.name}' has no adaptable compute entry"
            )
        ending = _matching_brace(code, match.end() - 1)
        invocation = match.group(1)
        body = code[match.end() : ending]
        body = re.sub(rf"\b{invocation}\.xy\b", "vec2<i32>(round(position))", body)
        # Hand-written field kernels use the named invocation's x/y separately.
        body = re.sub(rf"\b{invocation}\.([xy])\b", r"i32(round(position.\1))", body)
        body = body.replace("return;", "return vec4<f32>(0.0);")
        body = _replace_calls(
            body,
            "textureStore",
            lambda image, pos, value: (
                f"_spatial_result={value}"
                if image == kernel.output
                else "_spatial_result=_spatial_result"
            ),
        )
        body = (
            "var _spatial_result=vec4<f32>(0.0);\n" + body + "\nreturn _spatial_result;"
        )
        code = (
            code[: match.start()]
            + f"fn dispatch_local(position: vec2<f32>) -> vec4<f32> {{\n{body}\n}}"
            + code[ending + 1 :]
        )
        code = _IMAGE_BINDING.sub("", code)
        bindings = _UNIFORM_BINDING.findall(code)
        code = _UNIFORM_BINDING.sub("", code)
        for instance, _ in bindings:
            for uniform, expr in params.items():
                if expr.startswith("array<"):
                    code = re.sub(
                        rf"\b{instance}\.{uniform}\[([^\]]+)\](?:\.value)?",
                        lambda match: f"({expr})[{match[1]}]",
                        code,
                    )
                code = re.sub(
                    rf"\b{instance}\.{uniform}\b", lambda _: f"({expr})", code
                )
        code = _replace_calls(code, "textureDimensions", lambda _: "canvas_extent()")
        # Bounds checks in generated image reads address a baked canvas.
        # Inlined child shapes are callable on either side of the origin.
        code = re.sub(
            r"if\s*\(\(any\(\(p < vec2<i32>\(0\)\)\) \|\| any\(\(p >= n\)\)\)\)\s*"
            r"\{\s*return dualC\(0\.0\);\s*\}",
            "",
            code,
        )
        code = _replace_calls(
            code,
            "textureLoad",
            lambda image, pos: (
                f"{children[image]}(vec2<f32>({pos}) + fractional_position)"
            ),
        )
        # Primitive evaluators normally take an integer pixel. Preserve their
        # true local position when an enclosing transform produces fractions.
        code = re.sub(
            r"vec2<f32>\(pos\)", "(vec2<f32>(pos) + fractional_position)", code
        )
        code = "var<private> fractional_position: vec2<f32>;\n" + code
        code += """
fn invoke(position: vec2<f32>) -> vec4<f32> {
    fractional_position = position-vec2<f32>(round(position));
    return dispatch_local(position);
}
"""
        # Isolate the shader's helper functions, types and private variables.
        names = re.findall(r"\b(?:fn|struct|alias|const|var<private>)\s+(\w+)", code)
        # Function-local constants have a local declaration and may safely be renamed.
        # Coordinates, sampled accessors and the common bindings are generated above.
        renames = {var: f"{name}_{var}" for var in set(names)}
        renames["invoke"] = name
        return re.sub(r"\b\w+\b", lambda m: renames.get(m[0], m[0]), code)


def render_spatial(renderer, node):
    from ..core.field import FieldTexture
    from ..core.sdf import SDFTexture

    if node.kind not in {"sdf", "field", "color"}:
        raise ValueError("Select individual outputs before rendering a spatial shader")
    compiler = Compiler(renderer)
    target = compiler.evaluate(node)
    # Every value is a uniform; edits to its value don't rebuild the pipeline.
    size = renderer.ctx.size
    # Append the full extent at the end so preceding indices stay stable.
    width = compiler.constant(size[0])
    height = compiler.constant(size[1])
    ctx = renderer.ctx
    if node.kind == "sdf":
        allocate = ctx.r32f
    elif node.kind == "field":
        allocate = ctx.rgba32f
    else:
        allocate = ctx.rgba32f if node.mode == "LINEAR" else ctx.rgba8
    output = allocate()
    try:
        ox, oy = map(compiler.constant, output.origin)
        samplers = "\n".join(
            compiler.sampler(i, *item) for i, item in enumerate(compiler.maps)
        )
        count = (len(compiler.constants) + 3) // 4
        compiler.constants.extend([0.0] * (4 * count - len(compiler.constants)))
        source = f"""
struct SpatialValues {{ data: array<vec4<f32>, {count}>, }}
@group(0) @binding(0) var dest: texture_storage_2d<{output.format}, write>;
@group(0) @binding(1) var<uniform> values: SpatialValues;
var<private> world: vec2<f32>;
var<private> cell: vec2<f32>;
var<private> basis: mat2x2<f32>;
fn spatial_value(i: i32) -> f32 {{ return values.data[i/4][i%4]; }}
fn canvas_extent() -> vec2<u32> {{ return vec2<u32>(u32({width}), u32({height})); }}
"""
        source += samplers
        source += "\n".join(compiler.functions)
        source += f"""
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    if any(id.xy >= textureDimensions(dest)) {{ return; }}
    world=vec2<f32>(id.xy)+vec2<f32>({ox}, {oy});
    cell=world;
    basis=mat2x2<f32>(vec2<f32>(1,0),vec2<f32>(0,1));
    textureStore(dest, vec2<i32>(id.xy), {target}(world));
}}
"""
        contents = struct.pack(f"<{len(compiler.constants)}f", *compiler.constants)
        uniforms = ctx._uniform_buffer_for_payload(
            ("spatial", len(contents)), contents, label="spatial parameters"
        )
        inputs = {i + 2: image.tex for i, (_, image) in enumerate(compiler.maps)}
        if len(inputs) + 1 > ctx.device.limits["max-storage-textures-per-shader-stage"]:
            raise ValueError(
                "Spatial graph exceeds the adapter limit for distinct sampled textures"
            )
        ctx.run_shader(
            "spatial_parameters",
            {0: output, 1: uniforms, **inputs},
            source_code=source,
            group_count=((output.size[0] + 15) // 16, (output.size[1] + 15) // 16, 1),
        )
        if node.kind == "sdf":
            return SDFTexture(tex=output, context=ctx)
        if node.kind == "field":
            return FieldTexture(tex=output, context=ctx)
        return _color_output(node, output, ctx)
    except BaseException:
        output.close()
        raise


def _interval(renderer, value):
    """Conservative bounds for clamped numerical expressions used by layout."""
    if not is_spatial(value):
        resolved = renderer._resolve(value)
        if isinstance(resolved, Real) and math.isfinite(resolved):
            return float(resolved), float(resolved)
        return -math.inf, math.inf
    if not isinstance(value, Expr) or value.op == "texture_sample":
        return -math.inf, math.inf
    bounds = [_interval(renderer, item) for item in value.args]
    if value.op == "minimum":
        return min(item[0] for item in bounds), min(item[1] for item in bounds)
    if value.op in {"maximum", "relu"}:
        if value.op == "relu":
            bounds.append((0, 0))
        return max(item[0] for item in bounds), max(item[1] for item in bounds)
    if value.op == "neg":
        low, high = bounds[0]
        return -high, -low
    if value.op in {"sin", "cos", "tanh"}:
        return -1, 1
    if value.op == "sigmoid":
        return 0, 1
    if value.op == "add":
        result = bounds[0][0] + bounds[1][0], bounds[0][1] + bounds[1][1]
    elif value.op == "sub":
        result = bounds[0][0] - bounds[1][1], bounds[0][1] - bounds[1][0]
    elif value.op == "mul" and all(math.isfinite(v) for pair in bounds for v in pair):
        products = [a * b for a in bounds[0] for b in bounds[1]]
        result = min(products), max(products)
    elif value.op == "div" and bounds[1][0] * bounds[1][1] > 0:
        quotients = [a / b for a in bounds[0] for b in bounds[1]]
        result = min(quotients), max(quotients)
    else:
        return -math.inf, math.inf
    return (
        result
        if not any(math.isnan(item) for item in result)
        else (-math.inf, math.inf)
    )


def tile_dependencies(renderer, node):
    """Return map dependencies and each one's maximum sampling reach."""
    from ..core.spatial import samples

    result = {}

    def bounds(value):
        return max(abs(x) for x in _interval(renderer, value))

    def resident(current):
        image = getattr(current, "tex", None)
        if image is None:
            return False
        image._check_open()
        if image.context is not renderer.ctx:
            raise ValueError("Spatial textures must belong to the rendering canvas")
        if image.origin != (0, 0) or image.size != image.canvas_size:
            raise ValueError("Tiled spatial sampling requires a complete source image")
        return True

    def visit(current, reach, local_scale):
        if resident(current):
            return
        raw = dict(current.params)
        next_reach, next_scale = reach, local_scale
        if current.op == "spatial_repeat":
            if raw["baked"] and not resident(current.inputs[0]):
                raise NotImplementedError(
                    "Render a baked stamp once before using it in a tiled export"
                )
            spacing = raw["spacing"]
            if not isinstance(spacing, (list, tuple)):
                spacing = renderer._resolve(spacing)
            radius = 0.5 * math.hypot(*(bounds(component) for component in spacing))
            next_reach = radius * local_scale * renderer.ctx.device_scale
            next_scale = local_scale * bounds(raw["scale"])
        elif current.op == "scale":
            next_scale *= bounds(raw["factor"])
        elif current.op == "skew":
            shear = raw["skew"]
            if isinstance(shear, (tuple, list)):
                next_scale *= 1 + max(bounds(component) for component in shear)
            else:
                next_scale = math.inf
        elif current.op not in _TRANSFORMS | {"output"}:
            plugin = registry.get(current.op)
            if plugin.spatial_lowering is not None:
                visit(plugin.spatial_lowering(current.inputs, raw), reach, local_scale)
                return
            if plugin.shader is None and current.op not in _INLINE_COMPOSITES:
                raise NotImplementedError(
                    f"Tiled spatial pattern cannot inline '{current.op}'"
                )
        for name, value in current.params:
            for item in samples(value):
                texture, _, scope = item.args
                if resident(texture):
                    continue
                sample_reach = 0
                if scope == "cell":
                    sample_reach = (
                        next_reach
                        if current.op == "spatial_repeat"
                        and name in {"rotation", "scale", "stamp_center"}
                        else reach
                    )
                if not math.isfinite(sample_reach):
                    raise ValueError(
                        "Tiled per-cell sampling needs bounded spacing and scale"
                    )
                result[texture] = max(
                    result.get(texture, 0), math.ceil(sample_reach) + 2
                )
        for child in current.inputs:
            visit(child, next_reach, next_scale)

    visit(node, 0, 1)
    return tuple(result.items())
