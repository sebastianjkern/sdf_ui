__docformat__ = "google"

from pathlib import Path

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import lower_composite, shader

from .implementation import CONTRACT, INTERFACE, PROGRAM

_FUSED_OVERLAY = (
    Path(__file__).with_name("overlay_fused.wgsl").read_text(encoding="utf-8")
)
_BATCH_OVERLAY = (
    Path(__file__).with_name("overlay_batch.wgsl").read_text(encoding="utf-8")
)

MAX_BATCH_OVERLAYS = 3


def render_alpha_overlay(renderer, inputs, params):
    return render_alpha_overlay_batch(renderer, inputs[0], inputs[1:])[0]


def render_alpha_overlay_pair(renderer, inputs, params):
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


def render_alpha_overlay_batch(renderer, base, tops, *, output_indices=None):
    """Composite ordered overlays, materializing selected intermediate steps."""
    from sdf_ui.core.color import ColorSpaceMode, ColorTexture

    tops = tuple(tops)
    if not 1 <= len(tops) <= MAX_BATCH_OVERLAYS:
        raise ValueError("overlay batch size must be between one and three")
    output_indices = (
        tuple(range(len(tops)))
        if output_indices is None
        else tuple(sorted(set(output_indices)))
    )
    if not output_indices or any(index < 0 or index >= len(tops) for index in output_indices):
        raise ValueError("output_indices must select at least one overlay step")

    outputs = tuple(renderer.ctx.rgba8() for _ in output_indices)
    output_bindings = {
        stage: binding for binding, stage in enumerate(output_indices)
    }
    replacements = {
        "{{OUTPUT_BINDINGS}}": "\n".join(
            f"@group(0) @binding({binding})\n"
            f"var destTex{stage}: texture_storage_2d<rgba8unorm, write>;"
            for stage, binding in output_bindings.items()
        ),
        "{{BASE_BINDING}}": str(len(outputs)),
        "{{TOP_BINDINGS}}": "\n".join(
            f"@group(0) @binding({len(outputs) + index + 1})\n"
            f"var topTex{index}: texture_storage_2d<rgba8unorm, read>;"
            for index in range(len(tops))
        ),
        "const BASE_IS_LAB: bool = false;": (
            "const BASE_IS_LAB: bool = "
            + str(base.mode == ColorSpaceMode.LAB).lower()
            + ";"
        ),
        "{{TOP_FLAGS}}": "\n".join(
            f"const TOP{index}_IS_LAB: bool = "
            f"{str(top.mode == ColorSpaceMode.LAB).lower()};"
            for index, top in enumerate(tops)
        ),
        "{{STEPS}}": "\n".join(
            _batch_step(index, output_bindings.get(index))
            for index in range(len(tops))
        ),
    }
    source = _BATCH_OVERLAY
    for token, replacement in replacements.items():
        source = source.replace(token, replacement)

    resources = {binding: outputs[binding] for binding in range(len(outputs))}
    resources[len(outputs)] = base.tex
    resources.update(
        (len(outputs) + index + 1, top.tex) for index, top in enumerate(tops)
    )
    try:
        renderer.ctx.run_shader("overlay", resources, source_code=source)
    except BaseException:
        for output in outputs:
            output.close()
        raise
    return {
        stage: ColorTexture(
            tex=outputs[binding], context=renderer.ctx, mode=ColorSpaceMode.RGB
        )
        for stage, binding in output_bindings.items()
    }


def _batch_step(index, output_binding):
    store = (
        f"textureStore(destTex{index}, position, accum);"
        if output_binding is not None
        else ""
    )
    return f"""
    var top{index} = textureLoad(topTex{index}, position);
    if (TOP{index}_IS_LAB) {{
        top{index} = quantize_rgba8(vec4<f32>(lab_to_rgb(top{index}.rgb), top{index}.a));
    }}
    let alpha{index} = top{index}.a + (1.0 - top{index}.a) * accum.a;
    var rgb{index} = vec3<f32>(0.0);
    if (alpha{index} > 0.0) {{
        rgb{index} = (top{index}.rgb * top{index}.a + accum.rgb * (1.0 - top{index}.a) * accum.a) / alpha{index};
    }}
    accum = quantize_rgba8(vec4<f32>(rgb{index}, alpha{index}));
    {store}
""".strip()


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
            render_func=render_alpha_overlay_pair,
            lowering_func=lower_composite,
            method_of=(TextureKind.COLOR,),
        )
    )
