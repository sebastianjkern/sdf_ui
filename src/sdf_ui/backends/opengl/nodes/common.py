__docformat__ = "google"

from .base import ShaderFile


def lower_composite(builder, plugin, inputs, raw, values):
    """Lower a builtin composite through the shared typed graph builder."""
    return builder.composite(plugin, inputs, raw, values)


def shader(name, path, *, derivative_contract=None, gpu_derivative=None, program=None,
           interface=None):
    path = f"backends/opengl/nodes/{path}"
    return ShaderFile(name, path, derivative_contract, gpu_derivative, program, interface)


def build(name, *args, **kwargs):
    from sdf_ui.backends.opengl.nodes.registry import registry

    return registry.build(name, *args, **kwargs)


def params(*names):
    return lambda values: {name: values[name] for name in names}


def color(value):
    from sdf_ui.learning.autodiff import Tensor, as_tensor, stack

    if isinstance(value, Tensor) or (
        isinstance(value, (list, tuple)) and any(isinstance(c, Tensor) for c in value)
    ):
        channels = (
            value
            if isinstance(value, Tensor)
            else stack(tuple(as_tensor(c) for c in value))
        )
        if channels.shape in {(3,), (4,)} and (abs(channels.numpy()) > 1).any():
            return channels / 255
        return channels
    if isinstance(value, (list, tuple)):
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
    from sdf_ui.core.operations import run_shader
    from sdf_ui.backends.opengl.nodes.registry import registry

    ctx = renderer.ctx
    tex = ctx.rgba8()
    rgb_inputs = tuple(ensure_rgb(renderer, texture) for texture in inputs)

    descriptor = next(
        (item for item in registry.shader_files() if item.name == shader_name), None
    )
    interface = None if descriptor is None else descriptor.interface
    output = interface.output if interface is not None else None
    declared_inputs = (
        {binding.name: binding for binding in interface.inputs}
        if interface is not None else {}
    )
    shader_uniforms = {
        output.name if output is not None else "destTex":
        output.unit if output is not None else 0
    }
    shader_uniforms.update(uniforms or {})

    image_bindings = [
        (tex, output.unit, output.read, output.write)
        if output is not None else (tex, 0, False, True)
    ]
    for location, (uniform_name, texture) in enumerate(
        zip(image_uniforms, rgb_inputs), start=1
    ):
        binding = declared_inputs.get(uniform_name)
        if interface is not None and binding is None:
            raise ValueError(f"Shader '{shader_name}' has no input '{uniform_name}'")
        unit = location if binding is None else binding.unit
        shader_uniforms[uniform_name] = unit
        image_bindings.append((
            texture.tex, unit,
            True if binding is None else binding.read,
            False if binding is None else binding.write,
        ))

    run_shader(
        ctx,
        shader_name,
        uniforms=shader_uniforms,
        image_bindings=image_bindings,
    )
    return ColorTexture(tex=tex, context=ctx, mode=ColorSpaceMode.RGB)


def blur_pass(ctx, shader_name, dest, src):
    from sdf_ui.core.operations import run_shader
    from sdf_ui.backends.opengl.nodes.postprocessing.blur.implementation import shader_parts

    interface, _, _ = shader_parts(shader_name)
    output, source = interface.output, interface.inputs[0]

    run_shader(
        ctx,
        shader_name,
        uniforms={output.name: output.unit, source.name: source.unit},
        image_bindings=(
            (dest, output.unit, output.read, output.write),
            (src.tex if hasattr(src, "tex") else src, source.unit, source.read, source.write),
        ),
    )
