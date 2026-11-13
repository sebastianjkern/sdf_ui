__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import (
    color,
    ensure_rgb,
    lower_composite,
    shader,
)

from .implementation import CONTRACT, INTERFACE, PROGRAM


def render_fill_from_texture(renderer, inputs, params):
    from sdf_ui.core.color import ColorSpaceMode, ColorTexture
    from sdf_ui.gpu.nodes.registry import registry

    ctx = renderer.ctx
    sdf_texture = inputs[0]
    color_texture = ensure_rgb(renderer, inputs[1])
    tex = ctx.rgba8()

    descriptor = next(
        item for item in registry.shader_files() if item.name == "fill_from_texture"
    )
    try:
        ctx.run_node_shader(
            descriptor,
            tex,
            {"origTex": color_texture.tex, "sdf": sdf_texture.tex},
            {
                "background": color(params["background"]),
                "inflate": params["inflate"],
            },
        )
    except BaseException:
        tex.close()
        raise
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
                "fill_from_texture", "shading/fill_from_texture/shader.glsl",
                derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE,
            ),
            input_uniforms=("sdf", "origTex"),
            mode="RGB",
            render_func=render_fill_from_texture,
            lowering_func=lower_composite,
            method_of=(TextureKind.SDF,),
        )
    )
