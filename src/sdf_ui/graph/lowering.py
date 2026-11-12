"""Symbolic lowering of public texture DAGs to native shader passes.

Lowering performs no pixel evaluation, GL allocation, or parameter resolution.
Executors zip ``Pass.inputs`` with ``Pass.contract.inputs`` (GLSL declaration
order), and call ``Pass.uniforms(resolve)`` once per execution. ``resolve`` must
evaluate Exprs in the current parameter/context scope, retaining Tensor leaves
and normalizing hex colors. Image bindings are supplied by the executor, not by
the uniform closures. Plans retain symbolic parameter bindings, never a tape.
"""

from dataclasses import dataclass
from hashlib import sha256
from numbers import Real
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from ..core.expressions import Expr
from ..backends.opengl.nodes.base import ShaderFile
from ..backends.opengl.nodes.registry import registry
from ..core.texture import MultiOutputResult, TextureNode
from ..learning.autodiff import Tensor
from ..learning.render import DifferentiableRenderer, _tensor

__all__ = ["Port", "Pass", "GraphPlan", "lower_graph"]


@dataclass(frozen=True)
class Port:
    pass_index: int
    output_index: int = 0
    kind: str = "sdf"
    mode: Optional[str] = None


@dataclass(frozen=True)
class Pass:
    shader: ShaderFile
    inputs: Tuple[Port, ...]
    uniforms: Callable

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))

    @property
    def contract(self):
        return self.shader.differentiation


@dataclass(frozen=True)
class GraphPlan:
    passes: Tuple[Pass, ...]
    outputs: Mapping[str, Port]
    # Structural fingerprint, independent of current Tensor values. It is not
    # a cache key for results or for exchanging live bindings between plans.
    fingerprint: str = ""
    parameter_names: Tuple[str, ...] = ()
    sources: tuple = ()
    topology_controls: tuple = ()
    required_parameters: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "passes", tuple(self.passes))
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "topology_controls", tuple(self.topology_controls))
        object.__setattr__(self, "required_parameters", tuple(self.required_parameters))


def _snapshot(value):
    if isinstance(value, (tuple, list)):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _snapshot(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        value = value.copy()
        value.flags.writeable = False
    return value


def _resolve(value, resolve):
    """Also accommodate resolvers that handle only individual Expr leaves."""
    if isinstance(value, tuple):
        return tuple(_resolve(item, resolve) for item in value)
    if isinstance(value, Mapping):
        return {key: _resolve(item, resolve) for key, item in value.items()}
    return resolve(value)


def _color(value):
    # This helper only operates on three/four-channel uniforms, never pixels.
    return DifferentiableRenderer._color(value)


def _fixed_integer(value, name, minimum=0):
    if (
        isinstance(value, (Tensor, Expr, bool))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or int(value) != value
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be a fixed compile-time integer >= {minimum}; "
            "Tensor/Expr control selectors require rebuilding the graph with a literal"
        )
    return int(value)


_COLOR_UNIFORMS = frozenset(
    {
        "color",
        "background",
        "fgColor",
        "bgColor",
        "outline",
        "line_color",
    }
)


class _Lowerer:
    def __init__(self):
        self.passes = []
        self.sources = []
        self.topology_controls = []
        self.memo = {}
        self.visiting = set()
        self.conversions = {}
        self.parameter_names = set()
        self.required_parameters = set()
        self.structure = []
        # Composite descriptors come from builtin registration, so a custom
        # shader with the same name cannot replace a known calculation.
        self.shaders = {}
        for descriptor in registry.shader_files():
            if descriptor.differentiation is not None:
                self.shaders.setdefault(descriptor.name, descriptor)

    def signature(self, value):
        if isinstance(value, Expr):
            if value.op == "param":
                self.parameter_names.add(value.args[0])
                if value.args[1] is None:
                    self.required_parameters.add(value.args[0])
            return ("expr", value.op, self.signature(value.args))
        if isinstance(value, Tensor):
            name = getattr(value, "name", None)
            if name is not None:
                self.parameter_names.add(name)
            self.parameter_names.update(p.name for p in value._versions)
            return ("tensor", name, value.shape)
        if isinstance(value, Mapping):
            return tuple(
                (key, self.signature(val)) for key, val in sorted(value.items())
            )
        if isinstance(value, (tuple, list)):
            return tuple(self.signature(item) for item in value)
        if isinstance(value, np.ndarray):
            return ("array", value.dtype.str, value.shape, value.tobytes().hex())
        return value

    def emit(self, shader, inputs=None, uniforms=None, mode=None):
        descriptor = self.shaders[shader] if isinstance(shader, str) else shader
        contract = descriptor.differentiation
        if contract is None:
            raise ValueError(
                f"Shader '{descriptor.name}' has no differentiation contract"
            )
        inputs = {} if inputs is None else inputs
        expected = tuple(argument.name for argument in contract.inputs)
        if set(inputs) != set(expected):
            raise ValueError(
                f"Shader '{descriptor.name}' expects input ports {expected}"
            )
        ordered = tuple(inputs[name] for name in expected)
        for argument, port in zip(contract.inputs, ordered):
            kind = "sdf" if argument.kind == "r32f" else "color"
            if not isinstance(port, Port) or port.kind != kind:
                raise ValueError(
                    f"Shader '{descriptor.name}' input '{argument.name}' requires {kind}"
                )
        index = len(self.passes)
        self.passes.append(Pass(descriptor, ordered, uniforms or (lambda resolve: {})))
        return tuple(
            Port(
                index,
                i,
                "sdf" if arg.kind == "r32f" else "color",
                None if arg.kind == "r32f" else mode or "LAB",
            )
            for i, arg in enumerate(contract.outputs)
        )

    def convert(self, port, mode):
        if port.mode == mode:
            return port
        key = (port, mode)
        if key not in self.conversions:
            self.conversions[key] = self.emit(
                "to_rgb" if mode == "RGB" else "to_lab", {"origTex": port}, mode=mode
            )[0]
        return self.conversions[key]

    def fill(self, source, values):
        def uniforms(resolve):
            p = values(resolve)
            return {
                "color": _color(p["fg_color"]),
                "background": _color(p.get("bg_color", (0, 0, 0, 1))),
                "inflate": p.get("inflate", 0),
                "first": p.get("inner", -1.5),
                "second": p.get("outer", 0),
            }

        return self.emit("fill", {"sdf": source}, uniforms, "LAB")[0]

    def blur(self, source, base, n):
        n = _fixed_integer(n, "blur iteration count")
        for _ in range(n + 1):
            source = self.emit(
                f"blur_ver_{base}", {"origTex": source}, mode=source.mode
            )[0]
            source = self.emit(
                f"blur_hor_{base}", {"origTex": source}, mode=source.mode
            )[0]
        return source

    def node(self, node):
        if not isinstance(node, (TextureNode, MultiOutputResult)):
            raise TypeError(
                "Graph inputs must be TextureNode or MultiOutputResult instances"
            )
        key = id(node)
        if key in self.visiting:
            raise ValueError(f"Cycle detected in texture graph at '{node.op}'")
        if key in self.memo:
            return self.memo[key]
        if getattr(node, "tex", None) is not None:
            context = getattr(node, "context", None)
            if getattr(context, "_closed", False) or not callable(
                getattr(context, "create_graph_backend", None)
            ):
                raise ValueError(
                    "Materialized source textures require a live Canvas context"
                )
            if node.kind not in {"sdf", "color"}:
                raise ValueError(
                    "Materialized graph sources must be SDF or color textures"
                )
            index = len(self.sources)
            mode = None if node.kind == "sdf" else node.mode
            self.sources.append(node)
            port = Port(-index - 1, 0, node.kind, mode)
            self.memo[key] = port
            self.structure.append(
                (
                    "source",
                    node.kind,
                    mode,
                    node.tex.size,
                    node.tex.components,
                    node.tex.dtype,
                    node.label,
                )
            )
            return port
        self.visiting.add(key)
        plugin = registry.get(node.op)
        composite = plugin.lowering_func is not None
        if plugin.render_func is not None and not composite:
            raise ValueError(
                f"Plugin '{plugin.name}' has an unsupported custom render function; "
                "provide an explicit shader lowering (no CPU fallback)"
            )
        if not composite and (
            plugin.shader is None or plugin.shader.differentiation is None
        ):
            raise ValueError(
                f"Plugin '{plugin.name}' needs a ShaderFile with known differentiation metadata"
            )
        if len(node.inputs) != len(plugin.input_kinds):
            raise ValueError(f"Plugin '{plugin.name}' has invalid input count")
        for expected, value in zip(plugin.input_kinds, node.inputs):
            if getattr(value, "kind", None) != expected:
                raise ValueError(f"Plugin '{plugin.name}' requires {expected} inputs")
        inputs = tuple(self.node(value) for value in node.inputs)
        raw = _snapshot(dict(plugin.defaults or {}, **dict(node.params)))
        control_names = {
            "blur": ("base", "n"),
            "blur_9": ("n",),
            "blur_13": ("n",),
            "shadow": ("distance",),
            "ngon": ("sides",),
            "output": ("index",),
            "convex_polygon": ("points",),
        }.get(plugin.name, ())
        if control_names:
            self.topology_controls.append((plugin.name, control_names))
        self.structure.append((plugin.name, self.signature(raw)))

        # Snapshot the parameter *bindings*. Exprs and Tensor leaves stay symbolic.
        def values(resolve):
            return {key: _resolve(value, resolve) for key, value in raw.items()}

        if composite:
            result = plugin.lowering_func(self, plugin, inputs, raw, values)
        else:

            def uniforms(resolve):
                p = values(resolve)
                original = p.get("shade_background")
                if plugin.name == "light" and isinstance(original, Tensor):
                    p["shade_background"] = original.item()
                result = dict(plugin.make_uniforms(p)) if plugin.make_uniforms else {}
                if plugin.name == "light" and isinstance(original, Tensor):
                    result["shadeBackground"] = original
                radius = result.get("corner_radius")
                if isinstance(radius, Tensor) and radius.shape == ():
                    result["corner_radius"] = _tensor((radius,) * 4)
                for name in _COLOR_UNIFORMS.intersection(result):
                    result[name] = _color(result[name])
                return result

            if len(plugin.input_uniforms) != len(inputs):
                raise ValueError(
                    f"Plugin '{plugin.name}' has invalid shader input bindings"
                )
            mode = plugin.mode or next(
                (p.mode for p in inputs if isinstance(p, Port) and p.mode), "LAB"
            )
            ports = self.emit(
                plugin.shader, dict(zip(plugin.input_uniforms, inputs)), uniforms, mode
            )
            result = ports[0] if len(ports) == 1 else ports
        self.memo[key] = result
        self.visiting.remove(key)
        return result

    def composite(self, plugin, inputs, raw, values):
        name = plugin.name
        if name == "output":
            index = _fixed_integer(raw["index"], "output index")
            if not isinstance(inputs[0], tuple) or index >= len(inputs[0]):
                raise ValueError("output index does not select a multi-output port")
            result = inputs[0][index]
            if raw["kind"] != result.kind:
                raise ValueError("output kind does not match selected port")
            return result
        if name == "masked_union":
            return self.emit(
                "masked_union", dict(zip(("sdf0", "sdf1"), inputs)), mode="RGB"
            )
        if name in {"translate", "scale", "rotate", "skew"}:

            def uniforms(resolve):
                p = values(resolve)
                if name == "scale":
                    factor = _tensor(p["factor"]).item()
                    if not np.isfinite(factor) or abs(factor) < 1e-12:
                        raise ValueError("scale factor must be finite and non-zero")
                if name == "skew":
                    x, y = p["skew"]
                    det = _tensor(1 - x * y).item()
                    if not np.isfinite(det) or abs(det) < 1e-12:
                        raise ValueError("skew values produce a singular transform")
                return dict(
                    p,
                    transform_kind=("translate", "scale", "rotate", "skew").index(name),
                )

            return self.emit("transform", {"sdf": inputs[0]}, uniforms)[0]
        if name in {"mask", "layer_mask", "alpha_overlay", "multiply"}:
            rgb = tuple(self.convert(port, "RGB") for port in inputs)
            shader, bindings = {
                "mask": ("layer_mask", ("tex0", "tex1", "mask")),
                "layer_mask": ("layer_mask", ("tex0", "tex1", "mask")),
                "alpha_overlay": ("overlay", ("tex1", "tex0")),
                "multiply": ("multiply", ("mask1", "mask2")),
            }[name]
            return self.emit(shader, dict(zip(bindings, rgb)), mode="RGB")[0]
        if name == "fill_from_texture":

            def uniforms(resolve):
                p = values(resolve)
                return {"background": _color(p["background"]), "inflate": p["inflate"]}

            return self.emit(
                "fill_from_texture",
                {"sdf": inputs[0], "origTex": self.convert(inputs[1], "RGB")},
                uniforms,
                "RGB",
            )[0]
        if name == "generate_mask":

            def fill_values(resolve):
                p = values(resolve)
                return {
                    "fg_color": p["color1"],
                    "bg_color": p["color0"],
                    "inflate": p["inflate"],
                }

            return self.convert(self.fill(inputs[0], fill_values), "RGB")
        if name in {"blur_9", "blur_13", "blur"}:
            if name != "blur":
                return self.blur(inputs[0], 9 if name == "blur_9" else 13, raw["n"])
            base = _fixed_integer(raw["base"], "blur base")
            result = self.blur(
                self.convert(inputs[0], "RGB"), 13 if base == 13 else 9, raw["n"]
            )
            return self.convert(result, inputs[0].mode)
        if name == "shadow":

            def fill_values(resolve):
                return {
                    "fg_color": (0, 0, 0, 1),
                    "bg_color": (0, 0, 0, 0),
                    "inflate": values(resolve)["inflate"],
                }

            mask = self.convert(self.fill(inputs[0], fill_values), "RGB")
            blurred = self.blur(mask, 9, raw["distance"])
            return self.emit(
                "transparency",
                {"tex0": blurred},
                lambda resolve: {"alpha": values(resolve)["transparency"]},
                "RGB",
            )[0]
        if name == "radial_gradient":
            field = self.emit(
                "circle",
                uniforms=lambda resolve: {"offset": values(resolve)["a"], "radius": 0},
            )[0]

            def fill_values(resolve):
                p = values(resolve)
                return {
                    "fg_color": p["color1"],
                    "bg_color": p["color2"],
                    "inner": p["inner"],
                    "outer": p["outer"],
                }

            return self.fill(field, fill_values)
        if name == "linear_gradient":

            def geometry(resolve):
                p = values(resolve)
                a, b = _tensor(p["a"]), _tensor(p["b"])
                dx, dy = a[0] - b[0], a[1] - b[1]
                return a, dx, dy

            def line_uniforms(resolve):
                a, dx, dy = geometry(resolve)
                return {
                    "a": _tensor((a[0] + dx, a[1] - dy)),
                    "b": _tensor((a[0] - dx, a[1] + dy)),
                }

            field = self.emit("line", uniforms=line_uniforms)[0]

            def fill_values(resolve):
                p = values(resolve)
                _, dx, dy = geometry(resolve)
                return {
                    "fg_color": p["color1"],
                    "bg_color": p["color2"],
                    "inner": 0,
                    "outer": (dx * dx + dy * dy).sqrt(),
                }

            return self.fill(field, fill_values)
        if name in {"convex_polygon", "diamond", "rhombus", "parallelogram", "ngon"}:
            return self.polygon(name, raw, values)
        raise ValueError(f"Plugin '{name}' needs an explicit shader lowering")

    def polygon(self, name, raw, values):
        if name == "ngon":
            count = _fixed_integer(raw["sides"], "ngon sides", 3)
        elif name == "convex_polygon":
            points = raw["points"]
            if isinstance(points, Expr):
                raise ValueError(
                    "convex_polygon points need a fixed compile-time vertex count"
                )
            if isinstance(points, Tensor):
                count = points.shape[0] if points.shape else 0
            else:
                count = len(points)
            if count < 3:
                raise ValueError("convex_polygon requires at least 3 points")
        else:
            count = 4

        def vertices(resolve):
            p = values(resolve)
            if name == "convex_polygon":
                points = _tensor(p["points"])
            else:
                cx, cy = _tensor(p["center"])
                if name in {"diamond", "rhombus"}:
                    rx, ry = _tensor(p["radii"])
                    points = (
                        (cx, cy - ry),
                        (cx + rx, cy),
                        (cx, cy + ry),
                        (cx - rx, cy),
                    )
                elif name == "parallelogram":
                    w, h = _tensor(p["size"]) * 0.5
                    skew = p["skew"]
                    points = (
                        (cx - w + skew, cy - h),
                        (cx + w + skew, cy - h),
                        (cx + w - skew, cy + h),
                        (cx - w - skew, cy + h),
                    )
                else:
                    rotation, radius = _tensor(p["rotation"]), p["radius"]
                    angles = tuple(
                        rotation + 2 * np.pi * i / count for i in range(count)
                    )
                    points = tuple(
                        (cx + radius * a.cos(), cy + radius * a.sin()) for a in angles
                    )
                points = _tensor(points)
            if points.shape != (count, 2):
                raise ValueError(f"{name} points must have shape ({count}, 2)")
            area = (
                sum(
                    points[i, 0] * points[(i + 1) % count, 1]
                    - points[(i + 1) % count, 0] * points[i, 1]
                    for i in range(count)
                )
                * 0.5
            )
            signed_area = area.item()
            if abs(signed_area) < 1e-9:
                raise ValueError("convex_polygon points must not be collinear")
            return points, 1 if signed_area > 0 else -1

        planes = []
        for index in range(count):

            def uniforms(resolve, index=index):
                points, winding = vertices(resolve)
                edge = points[(index + 1) % count] - points[index]
                return {
                    "point": points[index],
                    "normal": _tensor((winding * edge[1], -winding * edge[0])),
                }

            planes.append(self.emit("half_plane", uniforms=uniforms)[0])
        result = planes[0]
        for plane in planes[1:]:
            result = self.emit("intersection", {"sdf0": result, "sdf1": plane})[0]
        return result


def lower_graph(outputs: Mapping[str, TextureNode]) -> GraphPlan:
    """Compile named public graphs into an immutable, topologically ordered plan.

    Shared node identities and inserted color conversions are lowered once.
    Select individual outputs from multi-output graphs before naming them.
    Blur iteration/base and output-index selectors must be compile-time literals.
    Direct Tensor bindings remain live; derived eager Tensors retain the usual
    version checks and should instead be expressed as Exprs for reusable plans.
    """
    if not isinstance(outputs, Mapping):
        raise TypeError("lower_graph expects a mapping of output names to TextureNodes")
    if any(
        not isinstance(name, str) or not isinstance(node, TextureNode)
        for name, node in outputs.items()
    ):
        raise TypeError("Graph outputs must map string names to TextureNodes")
    lowerer = _Lowerer()
    ports = {}
    for name, node in sorted(outputs.items()):
        port = lowerer.node(node)
        if not isinstance(port, Port):
            raise ValueError("Select a single output port before naming a graph output")
        ports[name] = port
    topology = tuple((p.shader.name, p.shader.path, p.inputs) for p in lowerer.passes)
    fingerprint = sha256(
        repr((lowerer.structure, topology, tuple(ports.items()))).encode()
    ).hexdigest()
    return GraphPlan(
        tuple(lowerer.passes),
        ports,
        fingerprint,
        tuple(sorted(lowerer.parameter_names)),
        tuple(lowerer.sources),
        tuple(lowerer.topology_controls),
        tuple(sorted(lowerer.required_parameters)),
    )
