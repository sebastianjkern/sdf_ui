__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import shader


def render_masked_union(renderer, inputs, params):
    from sdf_ui.core.color import ColorSpaceMode, ColorTexture
    from sdf_ui.core.operations import run_shader
    from sdf_ui.core.sdf import SDFTexture

    ctx = renderer.ctx
    tex = ctx.r32f()
    mask = ctx.rgba8()

    run_shader(
        ctx,
        "masked_union",
        uniforms={"destTex": 0, "maskTex": 1, "sdf0": 2, "sdf1": 3},
        image_bindings=(
            (tex, 0, False, True),
            (mask, 1, False, True),
            (inputs[0].tex, 2, True, False),
            (inputs[1].tex, 3, True, False),
        ),
    )

    return SDFTexture(tex=tex, context=ctx), ColorTexture(
        tex=mask, context=ctx, mode=ColorSpaceMode.RGB
    )


def register_plugins(registry):
    registry.register(
        Plugin(
            "masked_union",
            PluginFamily.PRIMITIVE,
            TextureKind.MULTI_OUTPUT,
            (TextureKind.SDF, TextureKind.SDF),
            shader=shader(
                "masked_union", "plugins/primitives/masked_union/shader.glsl"
            ),
            render_func=render_masked_union,
            method_of=(TextureKind.SDF,),
        )
    )
