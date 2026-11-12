__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.common import color, ensure_rgb, lower_composite, shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


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
            INTERFACE.output.name: INTERFACE.output.unit,
            INTERFACE.inputs[0].name: INTERFACE.inputs[0].unit,
            INTERFACE.inputs[1].name: INTERFACE.inputs[1].unit,
            "background": color(params["background"]),
            "inflate": params["inflate"],
        },
        image_bindings=(
            (tex, INTERFACE.output.unit, INTERFACE.output.read, INTERFACE.output.write),
            (color_texture.tex, INTERFACE.inputs[0].unit, INTERFACE.inputs[0].read, INTERFACE.inputs[0].write),
            (sdf_texture.tex, INTERFACE.inputs[1].unit, INTERFACE.inputs[1].read, INTERFACE.inputs[1].write),
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
