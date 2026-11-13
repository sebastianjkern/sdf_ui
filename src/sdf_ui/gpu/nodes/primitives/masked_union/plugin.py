__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import lower_composite, shader

from .implementation import CONTRACT, INTERFACE, PROGRAM


def render_masked_union(renderer, inputs, params):
    from sdf_ui.core.color import ColorSpaceMode, ColorTexture
    from sdf_ui.core.sdf import SDFTexture

    ctx = renderer.ctx
    tex = ctx.r32f()
    mask = ctx.rgba8()

    from sdf_ui.gpu.nodes.registry import registry

    descriptor = next(
        item for item in registry.shader_files() if item.name == "masked_union"
    )
    try:
        ctx.run_node_shader(
            descriptor,
            tex,
            tuple(texture.tex for texture in inputs),
            {},
            extra_outputs=(mask,),
        )
    except BaseException:
        tex.close()
        mask.close()
        raise

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
                "masked_union", "primitives/masked_union/shader.glsl",
                derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE,
            ),
            render_func=render_masked_union,
            lowering_func=lower_composite,
            method_of=(TextureKind.SDF,),
        )
    )
