"""Explicit lowering of public plugin conventions to shader derivative ports."""

import numpy as np

from ..core.plugins.registry import registry
from .autodiff import Tensor, stack
from .render import _lab_to_rgb, _rgb_to_lab, _tensor

_COMPOSITES = {
    "alpha_overlay",
    "multiply",
    "mask",
    "layer_mask",
    "fill_from_texture",
    "masked_union",
    "translate",
    "scale",
    "rotate",
    "skew",
    "blur_9",
    "blur_13",
    "blur",
    "shadow",
    "radial_gradient",
    "linear_gradient",
    "output",
}


def supports_plugin(plugin):
    if plugin.render_func is None:
        return plugin.shader is not None and plugin.shader.differentiation is not None
    return plugin.name in _COMPOSITES and plugin.render_func.__module__.startswith(
        "sdf_ui.core.plugins."
    )


def _shader(renderer, name, inputs=None, params=None):
    descriptors = {value.name: value for value in registry.shader_files()}
    contract = descriptors[name].differentiation
    if contract is None:
        raise ValueError(f"Shader '{name}' has no derivative contract")
    return contract.evaluate(renderer.size, inputs, params)


def _rgb(value, mode):
    return value if mode == "RGB" else _lab_to_rgb(value)


def _integer(value, name, minimum=0):
    value = value.item() if isinstance(value, Tensor) else value
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a fixed integer")
    if int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _blur(renderer, source, base, n):
    n = _integer(n, "blur iteration count")
    for _ in range(n + 1):
        source = _shader(renderer, f"blur_ver_{base}", {"origTex": source})
        source = _shader(renderer, f"blur_hor_{base}", {"origTex": source})
    return source


def _fill(renderer, source, foreground, background, inflate=0, first=-1.5, second=0):
    return _shader(
        renderer,
        "fill",
        {"sdf": source},
        {
            "color": renderer._color(foreground),
            "background": renderer._color(background),
            "inflate": inflate,
            "first": first,
            "second": second,
        },
    )


def evaluate_plugin(renderer, plugin, inputs, params, input_modes):
    """Return a Tensor/mode or tuple-of-Tensors/tuple-of-modes.

    Composite passes retain the host plugin's color and iteration conventions.
    No opaque custom render function is executed during differentiation.
    """
    if not supports_plugin(plugin):
        raise ValueError(f"Plugin '{plugin.name}' needs an explicit derivative adapter")
    name = plugin.name
    if name == "output":
        index = _integer(params["index"], "output index")
        return inputs[0][index], input_modes[0][index]
    if name == "masked_union":
        result = _shader(
            renderer, "masked_union", {"sdf0": inputs[0], "sdf1": inputs[1]}
        )
        return result, (None, "RGB")
    if name in {"translate", "scale", "rotate", "skew"}:
        uniforms = {
            "transform_kind": ("translate", "scale", "rotate", "skew").index(name)
        }
        uniforms.update(params)
        return _shader(renderer, "transform", {"sdf": inputs[0]}, uniforms), None
    if name in {"mask", "layer_mask", "multiply", "alpha_overlay"}:
        values = tuple(_rgb(value, mode) for value, mode in zip(inputs, input_modes))
        if name == "multiply":
            ports, shader = {"mask1": values[0], "mask2": values[1]}, "multiply"
        elif name == "alpha_overlay":
            ports, shader = {"tex1": values[0], "tex0": values[1]}, "overlay"
        else:
            ports, shader = (
                {"tex0": values[0], "tex1": values[1], "mask": values[2]},
                "layer_mask",
            )
        return _shader(renderer, shader, ports), "RGB"
    if name == "fill_from_texture":
        return _shader(
            renderer,
            "fill_from_texture",
            {
                "sdf": inputs[0],
                "origTex": _rgb(inputs[1], input_modes[1]),
            },
            {
                "inflate": params["inflate"],
                "background": renderer._color(params["background"]),
            },
        ), "RGB"
    if name in {"blur_9", "blur_13"}:
        return _blur(
            renderer, inputs[0], 9 if name == "blur_9" else 13, params["n"]
        ), input_modes[0]
    if name == "blur":
        base = _integer(params["base"], "blur base")
        base = 13 if base == 13 else 9
        result = _blur(renderer, _rgb(inputs[0], input_modes[0]), base, params["n"])
        return (
            (result, "RGB") if input_modes[0] == "RGB" else (_rgb_to_lab(result), "LAB")
        )
    if name == "shadow":
        mask = _lab_to_rgb(
            _fill(renderer, inputs[0], (0, 0, 0, 1), (0, 0, 0, 0), params["inflate"])
        )
        blurred = _blur(renderer, mask, 9, params["distance"])
        return _shader(
            renderer,
            "transparency",
            {"tex0": blurred},
            {"alpha": params["transparency"]},
        ), "RGB"
    if name == "radial_gradient":
        field = _shader(renderer, "circle", {}, {"offset": params["a"], "radius": 0})
        return _fill(
            renderer,
            field,
            params["color1"],
            params["color2"],
            first=params["inner"],
            second=params["outer"],
        ), "LAB"
    if name == "linear_gradient":
        a, b = _tensor(params["a"]), _tensor(params["b"])
        dx, dy = a[0] - b[0], a[1] - b[1]
        field = _shader(
            renderer,
            "line",
            {},
            {
                "a": stack((a[0] + dx, a[1] - dy)),
                "b": stack((a[0] - dx, a[1] + dy)),
            },
        )
        return _fill(
            renderer,
            field,
            params["color1"],
            params["color2"],
            first=0,
            second=(dx * dx + dy * dy).sqrt(),
        ), "LAB"
    uniform_params = params
    if name == "light" and isinstance(params["shade_background"], Tensor):
        # The display adapter casts this mode to float. Preserve its continuous
        # factor in the shader's height curve while choosing the same branch.
        uniform_params = dict(
            params, shade_background=params["shade_background"].item()
        )
    uniforms = (
        {} if plugin.make_uniforms is None else plugin.make_uniforms(uniform_params)
    )
    if name == "light":
        uniforms["shadeBackground"] = params["shade_background"]
    for argument in plugin.shader.differentiation.parameters:
        if (
            argument.name
            in {"color", "background", "fgColor", "bgColor", "outline", "line_color"}
            and argument.name in uniforms
        ):
            uniforms[argument.name] = renderer._color(uniforms[argument.name])
    ports = dict(zip(plugin.input_uniforms, inputs))
    result = plugin.shader.differentiation.evaluate(renderer.size, ports, uniforms)
    mode = plugin.mode or next((mode for mode in input_modes if mode), None)
    if plugin.result == "color":
        mode = mode or "LAB"
    return result, mode
