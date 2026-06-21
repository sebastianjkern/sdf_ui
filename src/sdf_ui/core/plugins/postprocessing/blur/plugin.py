__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import blur_pass, shader


def render_blur_9(renderer, inputs, params):
    from sdf_ui.core.color import ColorTexture
    from sdf_ui.core.context import decrease_tex_registry

    ctx = renderer.ctx
    tex0 = ctx.rgba8()
    tex1 = ctx.rgba8()

    blur_pass(ctx, "blur_ver_9", tex0, inputs[0])
    blur_pass(ctx, "blur_hor_9", tex1, tex0)

    for _ in range(params["n"]):
        blur_pass(ctx, "blur_ver_9", tex0, tex1)
        blur_pass(ctx, "blur_hor_9", tex1, tex0)

    tex0.release()
    decrease_tex_registry()
    return ColorTexture(tex=tex1, context=ctx, mode=inputs[0].mode)


def render_blur_13(renderer, inputs, params):
    from sdf_ui.core.color import ColorTexture
    from sdf_ui.core.context import decrease_tex_registry

    ctx = renderer.ctx
    tex0 = ctx.rgba8()
    tex1 = ctx.rgba8()

    blur_pass(ctx, "blur_ver_13", tex0, inputs[0])
    blur_pass(ctx, "blur_hor_13", tex1, tex0)

    for _ in range(params["n"]):
        blur_pass(ctx, "blur_ver_13", tex0, tex1)
        blur_pass(ctx, "blur_hor_13", tex1, tex0)

    tex0.release()
    decrease_tex_registry()
    return ColorTexture(tex=tex1, context=ctx, mode=inputs[0].mode)


def render_blur(renderer, inputs, params):
    from sdf_ui.core.plugins.registry import registry

    texture = inputs[0]
    rgb = (
        texture
        if texture.mode == "RGB"
        else renderer.render(registry.build("to_rgb", texture))
    )
    base = 13 if params["base"] == 13 else 9
    blurred = renderer.render(registry.build(f"blur_{base}", rgb, n=params["n"]))
    if texture.mode == "RGB":
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
                shader("blur_ver_9", "plugins/postprocessing/blur/blur9_vert.glsl"),
                shader("blur_hor_9", "plugins/postprocessing/blur/blur9_hor.glsl"),
            ),
            render_func=render_blur_9,
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
                shader("blur_ver_13", "plugins/postprocessing/blur/blur13_vert.glsl"),
                shader("blur_hor_13", "plugins/postprocessing/blur/blur13_hor.glsl"),
            ),
            render_func=render_blur_13,
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
            method_of=(TextureKind.COLOR,),
        )
    )
