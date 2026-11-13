__docformat__ = "google"

from .base import ShaderFile


def lower_composite(builder, plugin, inputs, raw, values):
    """Lower a builtin composite through the shared typed graph builder."""
    return builder.composite(plugin, inputs, raw, values)


def shader(
    name,
    path,
    *,
    derivative_contract=None,
    gpu_derivative=None,
    program=None,
    interface=None,
):
    path = f"gpu/nodes/{path}"
    return ShaderFile(
        name, path, derivative_contract, gpu_derivative, program, interface
    )


def build(name, *args, **kwargs):
    from sdf_ui.gpu.nodes.registry import registry

    return registry.build(name, *args, **kwargs)


def params(*names):
    return lambda values: {name: values[name] for name in names}


def color(value):
    from sdf_ui.learning.autodiff import Tensor, as_tensor, stack

    if _is_differentiable_color(value, Tensor):
        channels = _tensor_color_channels(value, Tensor, as_tensor, stack)
        if channels.shape in {(3,), (4,)} and (abs(channels.numpy()) > 1).any():
            return channels / 255
        return channels
    return _normalize_literal_color(value)


def _is_differentiable_color(value, tensor_type):
    return isinstance(value, tensor_type) or (
        isinstance(value, (list, tuple))
        and any(isinstance(channel, tensor_type) for channel in value)
    )


def _tensor_color_channels(value, tensor_type, as_tensor, stack):
    if isinstance(value, tensor_type):
        return value
    return stack(tuple(as_tensor(channel) for channel in value))


def _normalize_literal_color(value):
    if not isinstance(value, (list, tuple)):
        return value
    channels = tuple(value)
    if len(channels) in {3, 4} and any(abs(channel) > 1 for channel in channels):
        return tuple(channel / 255 for channel in channels)
    return value


def colors(*names):
    return lambda values: {name: color(values[name]) for name in names}


def ensure_rgb(renderer, texture):
    from sdf_ui.core.color import ColorSpaceMode

    if getattr(texture, "mode", None) == ColorSpaceMode.RGB:
        return texture
    return renderer.render(texture.to_rgb())


def render_rgb_shader(
    renderer, shader_name, inputs, *, uniforms=None, image_uniforms=()
):
    from sdf_ui.core.color import ColorSpaceMode, ColorTexture
    from sdf_ui.gpu.nodes.registry import registry

    ctx = renderer.ctx
    tex = ctx.rgba8()
    rgb_inputs = tuple(ensure_rgb(renderer, texture) for texture in inputs)

    descriptor = next(
        (item for item in registry.shader_files() if item.name == shader_name), None
    )
    interface = None if descriptor is None else descriptor.interface
    if descriptor is None or interface is None:
        raise ValueError(f"Shader '{shader_name}' has no declared image interface")
    names = tuple(image_uniforms) or tuple(
        binding.name for binding in interface.inputs
    )
    if len(names) != len(rgb_inputs):
        raise ValueError(f"Shader '{shader_name}' has invalid input count")
    try:
        ctx.run_node_shader(
            descriptor,
            tex,
            {name: texture.tex for name, texture in zip(names, rgb_inputs)},
            uniforms or {},
        )
    except BaseException:
        tex.close()
        raise
    return ColorTexture(tex=tex, context=ctx, mode=ColorSpaceMode.RGB)


def blur_pass(ctx, shader_name, dest, src):
    from sdf_ui.gpu.nodes.registry import registry

    descriptor = next(
        item for item in registry.shader_files() if item.name == shader_name
    )
    ctx.run_node_shader(
        descriptor,
        dest,
        (src.tex if hasattr(src, "tex") else src,),
        descriptor.differentiation.defaults or {},
    )
