"""Explicit lowering of public plugin conventions to shader derivative ports."""

from functools import partial

import numpy as np

from ...gpu.nodes.registry import registry
from ..autodiff import Tensor, stack
from ..render import _lab_to_rgb, _rgb_to_lab, _tensor

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
        "sdf_ui.gpu.nodes."
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


def _fill(renderer, source, foreground, background, inflate=0, first=-3 / 2, second=0):
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


class _PluginAdapters:
    @staticmethod
    def _adapt_output(renderer, plugin, inputs, params, modes):
        return _output_plugin(inputs, params, modes)

    @staticmethod
    def _adapt_masked_union(renderer, plugin, inputs, params, modes):
        return _masked_union_plugin(renderer, inputs, params)

    @staticmethod
    def _adapt_transform(renderer, plugin, inputs, params, modes):
        return _transform_plugin(renderer, plugin.name, inputs, params)

    @staticmethod
    def _adapt_layer(renderer, plugin, inputs, params, modes):
        return _layer_plugin(renderer, plugin.name, inputs, modes)

    @staticmethod
    def _adapt_fill_texture(renderer, plugin, inputs, params, modes):
        return _fill_texture_plugin(renderer, inputs, params, modes)

    @staticmethod
    def _adapt_fixed_blur(base, renderer, plugin, inputs, params, modes):
        return _blur(renderer, inputs[0], base, params["n"]), modes[0]

    @staticmethod
    def _adapt_blur(renderer, plugin, inputs, params, modes):
        return _iterated_blur_plugin(renderer, inputs, params, modes)

    @staticmethod
    def _adapt_shadow(renderer, plugin, inputs, params, modes):
        return _shadow_plugin(renderer, inputs, params)

    @staticmethod
    def _adapt_radial(renderer, plugin, inputs, params, modes):
        return _radial_gradient_plugin(renderer, params)

    @staticmethod
    def _adapt_linear(renderer, plugin, inputs, params, modes):
        return _linear_gradient_plugin(renderer, params)

    @staticmethod
    def result_mode(plugin, input_modes):
        mode = plugin.mode or next((value for value in input_modes if value), None)
        if plugin.result == "color":
            return mode or "LAB"
        return mode


def evaluate_plugin(renderer, plugin, inputs, params, input_modes):
    """Return differentiable results for a supported public plugin."""
    if not supports_plugin(plugin):
        raise ValueError(f"Plugin '{plugin.name}' needs an explicit derivative adapter")
    adapter = _PLUGIN_ADAPTERS.get(plugin.name)
    if adapter is None:
        return _shader_plugin(renderer, plugin, inputs, params, input_modes)
    return adapter(renderer, plugin, inputs, params, input_modes)


def _output_plugin(inputs, params, input_modes):
    index = _integer(params["index"], "output index")
    return inputs[0][index], input_modes[0][index]


def _masked_union_plugin(renderer, inputs, params):
    result = _shader(
        renderer,
        "masked_union",
        {"sdf0": inputs[0], "sdf1": inputs[1]},
        {"edgeWidth": params["edge_width"]},
    )
    return result, (None, "RGB")


def _transform_plugin(renderer, name, inputs, params):
    uniforms = {"transform_kind": ("translate", "scale", "rotate", "skew").index(name)}
    uniforms.update(params)
    return _shader(renderer, "transform", {"sdf": inputs[0]}, uniforms), None


def _layer_plugin(renderer, name, inputs, input_modes):
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


def _fill_texture_plugin(renderer, inputs, params, input_modes):
    result = _shader(
        renderer,
        "fill_from_texture",
        {"sdf": inputs[0], "origTex": _rgb(inputs[1], input_modes[1])},
        {
            "inflate": params["inflate"],
            "background": renderer._color(params["background"]),
        },
    )
    return result, "RGB"


def _iterated_blur_plugin(renderer, inputs, params, input_modes):
    base = _integer(params["base"], "blur base")
    base = 13 if base == 13 else 9
    mode = input_modes[0]
    source = inputs[0] if mode == "LINEAR" else _rgb(inputs[0], mode)
    result = _blur(renderer, source, base, params["n"])
    return (_rgb_to_lab(result), "LAB") if mode == "LAB" else (result, mode)


def _shadow_plugin(renderer, inputs, params):
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


def _radial_gradient_plugin(renderer, params):
    field = _shader(renderer, "circle", {}, {"offset": params["a"], "radius": 0})
    return _fill(
        renderer,
        field,
        params["color1"],
        params["color2"],
        first=params["inner"],
        second=params["outer"],
    ), "LAB"


def _linear_gradient_plugin(renderer, params):
    a, b = _tensor(params["a"]), _tensor(params["b"])
    dx, dy = a[0] - b[0], a[1] - b[1]
    field = _shader(
        renderer,
        "line",
        {},
        {"a": stack((a[0] + dx, a[1] - dy)), "b": stack((a[0] - dx, a[1] + dy))},
    )
    return _fill(
        renderer,
        field,
        params["color1"],
        params["color2"],
        first=0,
        second=(dx * dx + dy * dy).sqrt(),
    ), "LAB"


def _shader_uniform_parameters(plugin, params):
    if plugin.name == "light" and isinstance(params["shade_background"], Tensor):
        return dict(params, shade_background=params["shade_background"].item())
    return params


def _normalize_shader_colors(renderer, plugin, uniforms):
    color_names = {"color", "background", "fgColor", "bgColor", "outline", "line_color"}
    for argument in plugin.shader.differentiation.parameters:
        if argument.name in color_names and argument.name in uniforms:
            uniforms[argument.name] = renderer._color(uniforms[argument.name])


def _shader_plugin(renderer, plugin, inputs, params, input_modes):
    uniforms = (
        {}
        if plugin.make_uniforms is None
        else plugin.make_uniforms(_shader_uniform_parameters(plugin, params))
    )
    if plugin.name == "light":
        uniforms["shadeBackground"] = params["shade_background"]
    _normalize_shader_colors(renderer, plugin, uniforms)
    ports = dict(zip(plugin.input_uniforms, inputs))
    result = plugin.shader.differentiation.evaluate(renderer.size, ports, uniforms)
    return result, _PluginAdapters.result_mode(plugin, input_modes)


_PLUGIN_ADAPTERS = {
    "output": _PluginAdapters._adapt_output,
    "masked_union": _PluginAdapters._adapt_masked_union,
    "translate": _PluginAdapters._adapt_transform,
    "scale": _PluginAdapters._adapt_transform,
    "rotate": _PluginAdapters._adapt_transform,
    "skew": _PluginAdapters._adapt_transform,
    "mask": _PluginAdapters._adapt_layer,
    "layer_mask": _PluginAdapters._adapt_layer,
    "multiply": _PluginAdapters._adapt_layer,
    "alpha_overlay": _PluginAdapters._adapt_layer,
    "fill_from_texture": _PluginAdapters._adapt_fill_texture,
    "blur_9": partial(_PluginAdapters._adapt_fixed_blur, 9),
    "blur_13": partial(_PluginAdapters._adapt_fixed_blur, 13),
    "blur": _PluginAdapters._adapt_blur,
    "shadow": _PluginAdapters._adapt_shadow,
    "radial_gradient": _PluginAdapters._adapt_radial,
    "linear_gradient": _PluginAdapters._adapt_linear,
}
