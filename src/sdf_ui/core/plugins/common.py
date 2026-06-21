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


def blur_pass(ctx, shader_name, dest, src):
    from sdf_ui.core.operations import run_shader

    run_shader(
        ctx,
        shader_name,
        uniforms={"destTex": 0, "origTex": 1},
        image_bindings=((dest, 0, False, True), (src.tex if hasattr(src, "tex") else src, 1, True, False)),
    )
