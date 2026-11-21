__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import blur_pass, lower_composite, shader

from .implementation import shader_parts


def blur_shader(name, path):
    interface, contract = shader_parts(name)
    return shader(name, path, derivative_contract=contract, interface=interface)


def _render_blur_passes(renderer, inputs, params, base):
    from sdf_ui.core.color import ColorTexture

    ctx = renderer.ctx
    allocate = ctx.rgba32f if inputs[0].mode == "LINEAR" else ctx.rgba8
    scratch = allocate()
    output = None
    try:
        output = allocate()
        source = inputs[0]
        for _ in range(max(0, params["n"]) + 1):
            blur_pass(ctx, f"blur_ver_{base}", scratch, source)
            blur_pass(ctx, f"blur_hor_{base}", output, scratch)
            source = output
        return ColorTexture(tex=output, context=ctx, mode=inputs[0].mode)
    except BaseException:
        if output is not None:
            output.close()
        raise
    finally:
        scratch.close()


def render_blur_9(renderer, inputs, params):
    return _render_blur_passes(renderer, inputs, params, 9)


def render_blur_13(renderer, inputs, params):
    return _render_blur_passes(renderer, inputs, params, 13)


def render_blur(renderer, inputs, params):
    from sdf_ui.gpu.nodes.registry import registry

    texture = inputs[0]
    rgb = (
        texture
        if texture.mode in {"RGB", "LINEAR"}
        else renderer.render(registry.build("to_rgb", texture))
    )
    base = 13 if params["base"] == 13 else 9
    blurred = renderer.render(registry.build(f"blur_{base}", rgb, n=params["n"]))
    if texture.mode in {"RGB", "LINEAR"}:
        return blurred
    return renderer.render(registry.build("to_lab", blurred))


def register_plugins(registry):
    registry.register(
        Plugin(
            "blur_9",
            PluginFamily.POSTPROCESSING,
            TextureKind.COLOR,
            (TextureKind.COLOR,),
            params=("n",),
            defaults={"n": 0},
            extra_shaders=(
                blur_shader("blur_ver_9", "postprocessing/blur/blur_ver_9.wgsl"),
                blur_shader("blur_hor_9", "postprocessing/blur/blur_hor_9.wgsl"),
            ),
            render_func=render_blur_9,
            lowering_func=lower_composite,
            method_of=(TextureKind.COLOR,),
        )
    )
    registry.register(
        Plugin(
            "blur_13",
            PluginFamily.POSTPROCESSING,
            TextureKind.COLOR,
            (TextureKind.COLOR,),
            params=("n",),
            defaults={"n": 0},
            extra_shaders=(
                blur_shader("blur_ver_13", "postprocessing/blur/blur_ver_13.wgsl"),
                blur_shader("blur_hor_13", "postprocessing/blur/blur_hor_13.wgsl"),
            ),
            render_func=render_blur_13,
            lowering_func=lower_composite,
            method_of=(TextureKind.COLOR,),
        )
    )
    registry.register(
        Plugin(
            "blur",
            PluginFamily.POSTPROCESSING,
            TextureKind.COLOR,
            (TextureKind.COLOR,),
            params=("n", "base"),
            defaults={"n": 0, "base": 9},
            render_func=render_blur,
            lowering_func=lower_composite,
            method_of=(TextureKind.COLOR,),
        )
    )
