__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import lower_composite, shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def render_masked_union(renderer, inputs, params):
    from sdf_ui.core.color import ColorSpaceMode, ColorTexture
    from sdf_ui.core.operations import run_shader
    from sdf_ui.core.sdf import SDFTexture

    ctx = renderer.ctx
    tex = ctx.r32f()
    mask = ctx.rgba8()

    bindings = (INTERFACE.output, *INTERFACE.extra_outputs, *INTERFACE.inputs)
    textures = (tex, mask, inputs[0].tex, inputs[1].tex)
    run_shader(
        ctx,
        "masked_union",
        uniforms={binding.name: binding.unit for binding in bindings},
        image_bindings=tuple(
            (texture, binding.unit, binding.read, binding.write)
            for texture, binding in zip(textures, bindings)
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
                "masked_union", "plugins/primitives/masked_union/shader.glsl",
                derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE,
            ),
            render_func=render_masked_union,
            lowering_func=lower_composite,
            method_of=(TextureKind.SDF,),
        )
    )
