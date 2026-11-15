__docformat__ = "google"

from pathlib import Path

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import lower_composite, shader

from .implementation import CONTRACT, INTERFACE, PROGRAM

_FUSED_OVERLAY = (
    Path(__file__).with_name("overlay_fused.wgsl").read_text(encoding="utf-8")
)


def render_alpha_overlay(renderer, inputs, params):
    from sdf_ui.core.color import ColorSpaceMode, ColorTexture

    ctx = renderer.ctx
    output = ctx.rgba8()
    source = _FUSED_OVERLAY.replace(
        "const TOP_IS_LAB: bool = false;",
        "const TOP_IS_LAB: bool = "
        + str(inputs[1].mode == ColorSpaceMode.LAB).lower()
        + ";",
    ).replace(
        "const BASE_IS_LAB: bool = false;",
        "const BASE_IS_LAB: bool = "
        + str(inputs[0].mode == ColorSpaceMode.LAB).lower()
        + ";",
    )
    try:
        ctx.run_shader(
            "overlay",
            {0: output, 1: inputs[1].tex, 2: inputs[0].tex},
            source_code=source,
        )
    except BaseException:
        output.close()
        raise
    return ColorTexture(tex=output, context=ctx, mode=ColorSpaceMode.RGB)


def register_plugins(registry):
    registry.register(
        Plugin(
            "alpha_overlay",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.COLOR),
            shader=shader(
                "overlay",
                "layer/alpha_overlay/shader.glsl",
                derivative_contract=CONTRACT,
                program=PROGRAM,
                interface=INTERFACE,
            ),
            input_uniforms=("tex1", "tex0"),
            mode="RGB",
            render_func=render_alpha_overlay,
            lowering_func=lower_composite,
            method_of=(TextureKind.COLOR,),
        )
    )
