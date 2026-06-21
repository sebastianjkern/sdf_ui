__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import color, ensure_rgb, shader


def render_fill_from_texture(renderer, inputs, params):
    from sdf_ui.core.color import ColorSpaceMode, ColorTexture
    from sdf_ui.core.operations import run_shader

    ctx = renderer.ctx
    sdf_texture = inputs[0]
    color_texture = ensure_rgb(renderer, inputs[1])
    tex = ctx.rgba8()

    run_shader(
        ctx,
        "fill_from_texture",
        uniforms={
            "destTex": 0,
            "origTex": 1,
            "sdf": 2,
            "background": color(params["background"]),
            "inflate": params["inflate"],
        },
        image_bindings=(
            (tex, 0, False, True),
            (color_texture.tex, 1, True, False),
            (sdf_texture.tex, 2, True, False),
        ),
    )
    return ColorTexture(tex=tex, context=ctx, mode=ColorSpaceMode.RGB)


def register_plugins(registry):
    registry.register(
        Plugin(
            "fill_from_texture",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            (TextureKind.SDF, TextureKind.COLOR),
            params=("background", "inflate"),
            defaults={"background": (0.0, 0.0, 0.0, 0.0), "inflate": 0},
            shader=shader(
                "fill_from_texture", "plugins/shading/fill_from_texture/shader.glsl"
            ),
            input_uniforms=("sdf", "origTex"),
            mode="RGB",
            render_func=render_fill_from_texture,
            method_of=(TextureKind.SDF,),
        )
    )
