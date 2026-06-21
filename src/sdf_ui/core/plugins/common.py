__docformat__ = "google"

from .base import ShaderFile


def shader(name, path):
    if path.startswith("plugins/"):
        path = f"core/{path}"
    return ShaderFile(name, path)


def build(name, *args, **kwargs):
    from sdf_ui.core.plugins.registry import registry

    return registry.build(name, *args, **kwargs)


def params(*names):
    return lambda values: {name: values[name] for name in names}


def color(value):
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

    ctx = renderer.ctx
    tex = ctx.rgba8()
    rgb_inputs = tuple(ensure_rgb(renderer, texture) for texture in inputs)

    shader_uniforms = {"destTex": 0}
    shader_uniforms.update(uniforms or {})

    image_bindings = [(tex, 0, False, True)]
    for location, (uniform_name, texture) in enumerate(
        zip(image_uniforms, rgb_inputs), start=1
    ):
        shader_uniforms[uniform_name] = location
        image_bindings.append((texture.tex, location, True, False))

    run_shader(
        ctx,
        shader_name,
        uniforms=shader_uniforms,
        image_bindings=image_bindings,
    )
    return ColorTexture(tex=tex, context=ctx, mode=ColorSpaceMode.RGB)


def rgba_array(texture):
    import numpy as np

    width, height = texture.tex.size
    return np.frombuffer(texture.tex.read(), dtype=np.uint8).reshape((height, width, 4))


def rgb_texture_from_array(ctx, pixels):
    import numpy as np

    from sdf_ui.core.color import ColorSpaceMode, ColorTexture

    tex = ctx.rgba8()
    data = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
    tex.write(data.tobytes())
    return ColorTexture(tex=tex, context=ctx, mode=ColorSpaceMode.RGB)


def blur_pass(ctx, shader_name, dest, src):
    from sdf_ui.core.operations import run_shader

    run_shader(
        ctx,
        shader_name,
        uniforms={"destTex": 0, "origTex": 1},
        image_bindings=(
            (dest, 0, False, True),
            (src.tex if hasattr(src, "tex") else src, 1, True, False),
        ),
    )
