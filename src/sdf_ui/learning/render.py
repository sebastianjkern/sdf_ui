"""CPU NumPy rendering with reverse-mode gradients through a supported graph.

This is a floating-point reference backend, not GPU shader autograd. SDFs use
integer pixel coordinates and colors retain the shader API's normalized LAB
encoding until explicitly converted to RGB. Unlike rgba8 GPU intermediates,
values are never quantized or clipped during rendering.

Use ``param('radius')`` expressions in persistent scenes and bind Parameters
in the renderer's ``params`` mapping. Expressions are evaluated anew on every
render. Direct Parameters also work; derived eager Tensor expressions must be
rebuilt after updates. Each render owns its DAG memo and execution tape.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from ..backends.opengl.nodes.registry import registry
from ..core.color import ColorSpaceMode
from ..core.expressions import Expr, evaluate_expr
from ..core.texture import MultiOutputResult, TextureNode
from ..util import hex_col
from .autodiff import Tensor, as_tensor, maximum, minimum, stack, where


class UnsupportedGradientError(ValueError):
    """An operation has no differentiable implementation in this backend."""


def _tensor(value):
    # np.asarray on a container of Tensors would silently detach its leaves.
    if isinstance(value, (tuple, list)):
        return stack(tuple(_tensor(item) for item in value))
    return as_tensor(value)


def _clip(value, low, high):
    return minimum(maximum(value, low), high)


def _mix(first, second, weight):
    return first * (1.0 - weight) + second * weight


def _smoothstep(first, second, value):
    weight = _clip((value - first) / (second - first), 0.0, 1.0)
    return weight * weight * (3.0 - 2.0 * weight)


def _length(x, y):
    squared = x * x + y * y
    positive = squared.data > 0.0
    # Avoid sqrt'(0) even in the inactive branch of where; the origin uses
    # the zero subgradient of the Euclidean norm.
    return where(positive, where(positive, squared, 1.0).sqrt(), 0.0)


def _matrix(value, rows):
    return stack(
        tuple(
            sum(value[..., j] * coefficient for j, coefficient in enumerate(row))
            for row in rows
        ),
        axis=-1,
    )


def _rgba(rgb, alpha):
    return stack((rgb[..., 0], rgb[..., 1], rgb[..., 2], alpha), axis=-1)


def _rgb_to_lab(color):
    rgb = color[..., :3]
    linear = where(
        rgb.data > 0.04045,
        (maximum((rgb + 0.055) / 1.055, 1e-30)) ** 2.4,
        rgb / 12.92,
    )
    xyz = _matrix(
        linear,
        (
            (0.4124, 0.3576, 0.1805),
            (0.2126, 0.7152, 0.0722),
            (0.0193, 0.1192, 0.9505),
        ),
    ) / np.array((0.95047, 1.0, 1.08883))
    curved = where(
        xyz.data > 0.008856,
        maximum(xyz, 1e-30) ** (1.0 / 3.0),
        7.787 * xyz + 16.0 / 116.0,
    )
    return stack(
        (
            (116.0 * curved[..., 1] - 16.0) / 100.0,
            0.5 + 500.0 * (curved[..., 0] - curved[..., 1]) / 254.0,
            0.5 + 200.0 * (curved[..., 1] - curved[..., 2]) / 254.0,
            color[..., 3],
        ),
        axis=-1,
    )


def _lab_to_rgb(color):
    fy = (100.0 * color[..., 0] + 16.0) / 116.0
    fx = 254.0 * (color[..., 1] - 0.5) / 500.0 + fy
    fz = fy - 254.0 * (color[..., 2] - 0.5) / 200.0
    curved = stack((fx, fy, fz), axis=-1)
    xyz = where(
        curved.data > 0.206897,
        curved**3,
        (curved - 16.0 / 116.0) / 7.787,
    ) * np.array((0.95047, 1.0, 1.08883))
    linear = _matrix(
        xyz,
        (
            (3.2406, -1.5372, -0.4986),
            (-0.9689, 1.8758, 0.0415),
            (0.0557, -0.2040, 1.0570),
        ),
    )
    rgb = where(
        linear.data > 0.0031308,
        1.055 * maximum(linear, 1e-30) ** (1.0 / 2.4) - 0.055,
        12.92 * linear,
    )
    return _rgba(rgb, color[..., 3])


class DifferentiableRenderer:
    """Render existing TextureNodes to float Tensors without allocating a GPU.

    ``size`` is ``(width, height)``. ``render`` returns ``(height, width)``
    SDFs or ``(height, width, 4)`` straight-alpha colors, with row zero at the
    bottom. Builtins implement their GLSL formulas, including LAB interpolation,
    rectangle half-extents, and clamped bilinear SDF transforms. Branches use
    piecewise derivatives; hard CSG boundaries are not globally smooth.

    Plugins may provide ``differentiable_func(renderer, inputs, params)`` that
    returns a Tensor made using autodiff math. Color mode follows plugin.mode,
    then the first color input, then LAB. The entire graph is checked before
    running any hook. GPU-only and materialized textures fail explicitly.
    """

    _operations = frozenset(
        {
            "circle",
            "disc",
            "rect",
            "rounded_rect",
            "union",
            "intersection",
            "subtract",
            "smooth_union",
            "interpolate",
            "abs",
            "fill",
            "clear",
            "clear_color",
            "to_rgb",
            "to_lab",
            "alpha_overlay",
            "transparency",
            "multiply",
            "translate",
            "scale",
            "rotate",
            "generate_mask",
        }
    )

    def __init__(self, size, params=None):
        if len(size) != 2 or any(
            isinstance(value, bool)
            or not np.isfinite(value)
            or int(value) != value
            or value <= 0
            for value in size
        ):
            raise ValueError("size must contain positive integer width and height")
        self.size = tuple(int(value) for value in size)
        self.params = {} if params is None else params
        self.ctx = self  # expression context; no OpenGL Context is constructed
        y, x = np.indices((self.size[1], self.size[0]), dtype=float)
        self.x, self.y = as_tensor(x), as_tensor(y)

    def percent(self, value):
        return self.percent_x(value)

    def percent_x(self, value):
        return value / 100.0 * self.size[0]

    def percent_y(self, value):
        return value / 100.0 * self.size[1]

    def percent_of_min(self, value):
        return value / 100.0 * min(self.size)

    @classmethod
    def supported_operations(cls):
        """Return sorted builtin and registered differentiable operation names.

        The tuple reflects the current plugin registry, including private
        plugins with a differentiable hook. Discovery does not execute hooks.
        """
        from .shader_plugins import supports_plugin

        registry.ensure_loaded()
        custom = {
            name
            for name, plugin in registry._plugins.items()
            if getattr(plugin, "differentiable_func", None) is not None
            or supports_plugin(plugin)
        }
        return tuple(sorted(cls._operations | custom))

    def render(self, scene):
        """Build a fresh tape, sharing each DAG node only within this call."""
        return self._render(scene)[0]

    def render_plan(self, plan):
        """Evaluate all named outputs of a shared graph plan on the CPU."""
        from ..backends.plan import execute_reference

        return execute_reference(plan, self)

    def _render(self, scene):
        if not isinstance(scene, TextureNode):
            raise TypeError("DifferentiableRenderer.render expects a TextureNode")
        plugins = {}
        self._validate(scene, plugins, set())
        memo, modes = {}, {}
        result = self._eval(scene, plugins, memo, modes)
        return result, modes[id(scene)]

    def _validate(self, value, plugins, visiting):
        if isinstance(value, (TextureNode, MultiOutputResult)):
            self._validate_node(value, plugins, visiting)
            return
        if isinstance(value, Expr):
            self._validate_expression(value, plugins, visiting)
            return
        if isinstance(value, dict):
            self._validate_values(value.values(), plugins, visiting)
        elif isinstance(value, (tuple, list)):
            self._validate_values(value, plugins, visiting)

    def _validate_node(self, value, plugins, visiting):
        key = id(value)
        if key in visiting:
            raise ValueError("Cycle in differentiable render graph")
        if key in plugins:
            return
        plugin = self._node_plugin(value)
        visiting.add(key)
        try:
            self._validate_values(value.inputs, plugins, visiting)
            self._validate_values((item for _, item in value.params), plugins, visiting)
        finally:
            visiting.remove(key)
        plugins[key] = plugin

    def _node_plugin(self, value):
        from .shader_plugins import supports_plugin

        op = value.op
        if getattr(value, "tex", None) is not None:
            raise UnsupportedGradientError(
                f"Unsupported gradient operation '{op or 'materialized texture'}'"
            )
        try:
            plugin = registry.get(op)
        except KeyError as exc:
            raise UnsupportedGradientError(
                f"Unsupported gradient operation '{op}'"
            ) from exc
        if (
            op not in self._operations
            and getattr(plugin, "differentiable_func", None) is None
            and op != "output"
            and not supports_plugin(plugin)
        ):
            raise UnsupportedGradientError(f"Unsupported gradient operation '{op}'")
        return plugin

    def _validate_expression(self, value, plugins, visiting):
        if value.op == "param":
            name, default = value.args
            self._validate(self.params.get(name, default), plugins, visiting)
            return
        self._validate_values(value.args, plugins, visiting)

    def _validate_values(self, values, plugins, visiting):
        for item in values:
            self._validate(item, plugins, visiting)

    def _resolve(self, value, plugins, memo, modes):
        if isinstance(value, (Expr, str)):
            value = evaluate_expr(value, self.ctx, self.params, track_grad=True)
        if isinstance(value, (TextureNode, MultiOutputResult)):
            return self._eval(value, plugins, memo, modes)
        if isinstance(value, (tuple, list)):
            return tuple(self._resolve(item, plugins, memo, modes) for item in value)
        if isinstance(value, dict):
            return {
                key: self._resolve(item, plugins, memo, modes)
                for key, item in value.items()
            }
        if isinstance(value, str) and value.startswith("#"):
            return hex_col(value)
        return value

    def _eval(self, node, plugins, memo, modes):
        key = id(node)
        if key in memo:
            return memo[key]
        plugin = plugins[key]
        inputs = tuple(
            self._resolve(item, plugins, memo, modes) for item in node.inputs
        )
        params = dict(plugin.defaults or {})
        params.update(
            {
                name: self._resolve(value, plugins, memo, modes)
                for name, value in node.params
            }
        )
        input_modes = tuple(modes.get(id(item)) for item in node.inputs)
        mode = plugin.mode or next((item for item in input_modes if item), None)
        if node.kind == "color":
            mode = mode or ColorSpaceMode.LAB
        result, mode = self._evaluate_node_operation(
            node, plugin, inputs, params, input_modes, mode
        )
        if node.kind == "multi_output":
            if not isinstance(result, tuple) or not all(
                isinstance(value, Tensor) for value in result
            ):
                raise TypeError(
                    "Multi-output derivative must return a tuple of Tensors"
                )
            memo[key], modes[key] = result, mode
            return result
        expected = (self.size[1], self.size[0])
        if node.kind == "color":
            expected += (4,)
        if result.shape != expected:
            raise ValueError(
                f"Differentiable operation '{node.op}' returned shape {result.shape}; "
                f"expected {expected}"
            )
        memo[key], modes[key] = result, mode
        return result

    def _evaluate_node_operation(self, node, plugin, inputs, params, input_modes, mode):
        hook = getattr(plugin, "differentiable_func", None)
        if node.op == "output":
            index = params["index"]
            if (
                isinstance(index, Tensor)
                or isinstance(index, bool)
                or not isinstance(index, int)
            ):
                raise ValueError("Multi-output index must be a fixed integer")
            if not isinstance(inputs[0], tuple) or not 0 <= index < len(inputs[0]):
                raise ValueError("Invalid multi-output selection")
            result = inputs[0][index]
            mode = input_modes[0][index]
        elif hook is not None:
            result = hook(self, inputs, params)
            if not isinstance(result, Tensor):
                raise TypeError(f"Differentiable hook '{node.op}' must return a Tensor")
        elif node.op not in self._operations:
            from .shader_plugins import evaluate_plugin

            result, mode = evaluate_plugin(self, plugin, inputs, params, input_modes)
        else:
            result = self._builtin(node.op, inputs, params, input_modes)
            mode = self._builtin_mode(node.op, mode)
        return result, mode

    @staticmethod
    def _builtin_mode(op, mode):
        if op in {"fill", "clear", "clear_color", "to_lab"}:
            return ColorSpaceMode.LAB
        if op in {"to_rgb", "generate_mask", "alpha_overlay", "multiply"}:
            return ColorSpaceMode.RGB
        return mode

    @staticmethod
    def _color(value):
        color = _tensor(value)
        if color.shape not in {(3,), (4,)}:
            raise ValueError("Colors must have three or four channels")
        # Match the existing tuple-color shorthand (0..255 when any channel
        # exceeds 1), extending it to trainable vectors without detaching them.
        if np.any(np.abs(color.data) > 1.0):
            color = color / 255.0
        if color.shape == (3,):
            color = stack((color[0], color[1], color[2], as_tensor(1.0)))
        return color

    def _fill(self, distance, params):
        weight = _smoothstep(
            _tensor(params.get("inner", -1.5)),
            _tensor(params.get("outer", 0.0)) + 0.0001,
            distance - _tensor(params.get("inflate", 0.0)),
        )[..., None]
        return _mix(
            _rgb_to_lab(self._color(params["fg_color"])),
            _rgb_to_lab(self._color(params["bg_color"])),
            weight,
        )

    def _builtin(self, op, inputs, params, modes):
        match op:
            case "circle" | "disc":
                return self._circle(params)
            case "rect" | "rounded_rect":
                return self._rectangle(params)
            case "union":
                return minimum(*inputs)
            case "intersection":
                return maximum(*inputs)
            case "subtract":
                return maximum(inputs[0], -inputs[1])
            case "smooth_union":
                return self._smooth_union(inputs, params)
            case "interpolate":
                return _mix(inputs[0], inputs[1], _tensor(params["t"]))
            case "abs":
                return inputs[0].abs()
            case "translate" | "scale" | "rotate":
                return self._transform(op, inputs[0], params)
            case "clear" | "clear_color":
                return self._clear(params)
            case "fill":
                return self._fill(inputs[0], params)
            case "generate_mask":
                return self._generate_mask(inputs, params)
            case "to_rgb":
                return _lab_to_rgb(inputs[0])
            case "to_lab":
                return _rgb_to_lab(inputs[0])
            case "transparency":
                return _rgba(
                    inputs[0][..., :3], inputs[0][..., 3] * _tensor(params["alpha"])
                )
            case "multiply" | "alpha_overlay":
                return self._layer_operation(op, inputs, modes)
            case _:
                raise UnsupportedGradientError(f"Unsupported gradient operation '{op}'")

    def _circle(self, params):
        center = _tensor(params["center"])
        return _length(self.x - center[0], self.y - center[1]) - _tensor(
            params["radius"]
        )

    def _rectangle(self, params):
        center, size = _tensor(params["center"]), _tensor(params["size"])
        angle = _tensor(params.get("angle", 0.0))
        x, y = self.x - center[0], self.y - center[1]
        u = x * angle.cos() - y * angle.sin()
        v = x * angle.sin() + y * angle.cos()
        radius = self._corner_radius(params, u, v)
        qx, qy = u.abs() - size[0] + radius, v.abs() - size[1] + radius
        return (
            minimum(maximum(qx, qy), 0.0)
            + _length(maximum(qx, 0.0), maximum(qy, 0.0))
            - radius
        )

    @staticmethod
    def _corner_radius(params, u, v):
        radius = _tensor(params.get("corner_radius", 0.0))
        if radius.shape == (4,):
            hx = _smoothstep(0.0, 0.0001, u)
            hy = _smoothstep(0.0, 0.0001, v)
            return _mix(
                _mix(radius[0], radius[2], hx), _mix(radius[1], radius[3], hx), hy
            )
        if radius.shape != ():
            raise ValueError("corner_radius must be a scalar or four radii")
        return radius

    @staticmethod
    def _smooth_union(inputs, params):
        k = maximum(_tensor(params["k"]), 1e-6)
        weight = _clip(0.5 + 0.5 * (inputs[1] - inputs[0]) / k, 0.0, 1.0)
        return _mix(inputs[1], inputs[0], weight) - k * weight * (1.0 - weight)

    def _clear(self, params):
        return _rgb_to_lab(self._color(params["color"])) + np.zeros(
            (self.size[1], self.size[0], 4)
        )

    def _generate_mask(self, inputs, params):
        return _lab_to_rgb(
            self._fill(
                inputs[0],
                {
                    "fg_color": params["color1"],
                    "bg_color": params["color0"],
                    "inflate": params["inflate"],
                },
            )
        )

    def _layer_operation(self, op, inputs, modes):
        rgb_inputs = tuple(
            _lab_to_rgb(value) if mode != ColorSpaceMode.RGB else value
            for value, mode in zip(inputs, modes)
        )
        if op == "multiply":
            return rgb_inputs[0] * rgb_inputs[1]
        return self._alpha_overlay(*rgb_inputs)

    @staticmethod
    def _alpha_overlay(base, top):
        alpha = top[..., 3] + (1.0 - top[..., 3]) * base[..., 3]
        positive = alpha.data > 0.0
        numerator = (
            top[..., :3] * top[..., 3:4]
            + base[..., :3] * (1.0 - top[..., 3:4]) * base[..., 3:4]
        )
        rgb = where(
            positive[..., None], numerator / where(positive, alpha, 1.0)[..., None], 0.0
        )
        return _rgba(rgb, alpha)

    def _transform(self, op, source, params):
        distance_scale = 1.0
        if op == "translate":
            offset = _tensor(params["offset"])
            x, y = self.x - offset[0], self.y - offset[1]
        else:
            center = _tensor(params["center"])
            x, y = self.x - center[0], self.y - center[1]
            if op == "scale":
                factor = _tensor(params["factor"])
                if (
                    factor.shape != ()
                    or not np.isfinite(factor.data)
                    or abs(factor.data) < 1e-12
                ):
                    raise ValueError("scale factor must be finite and non-zero")
                x, y = x / factor, y / factor
                distance_scale = factor.abs()
            else:
                angle = _tensor(params["angle"])
                x, y = (
                    angle.cos() * x + angle.sin() * y,
                    -angle.sin() * x + angle.cos() * y,
                )
            x, y = x + center[0], y + center[1]
        x, y = _clip(x, 0.0, self.size[0] - 1), _clip(y, 0.0, self.size[1] - 1)
        x0, y0 = np.floor(x.data).astype(int), np.floor(y.data).astype(int)
        x1, y1 = (
            np.minimum(x0 + 1, self.size[0] - 1),
            np.minimum(y0 + 1, self.size[1] - 1),
        )
        tx, ty = x - x0, y - y0
        return (
            _mix(
                _mix(source[y0, x0], source[y0, x1], tx),
                _mix(source[y1, x0], source[y1, x1], tx),
                ty,
            )
            * distance_scale
        )

    def save(self, scene, path):
        """Save display RGBA with RGB conversion and flip outside gradients.

        Returns the output Path. SDFs should first be shaded with ``fill`` or
        ``generate_mask``. No parameter gradients are changed by saving.
        """
        if not isinstance(scene, TextureNode) or scene.kind != "color":
            raise ValueError("save expects a color scene; shade SDFs with fill first")
        result, mode = self._render(scene)
        display = as_tensor(np.array(result.data, copy=True))
        if mode != ColorSpaceMode.RGB:
            display = _lab_to_rgb(display)
        pixels = np.flipud(
            np.clip(np.rint(display.data * 255.0), 0, 255).astype(np.uint8)
        )
        path = Path(path)
        Image.fromarray(pixels).save(path)
        return path
