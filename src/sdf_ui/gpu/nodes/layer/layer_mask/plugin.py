__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import (
    lower_composite,
    render_rgb_shader,
    shader,
)

from .implementation import CONTRACT, INTERFACE

_BLEND_MODES = {
    "mix": 0.0,
    "multiply": 1.0,
    "screen": 2.0,
    "over": 3.0,
    "darken": 4.0,
    "lighten": 5.0,
}


def _mode_uniform(params):
    try:
        return {"mode": _BLEND_MODES[params["mode"]]}
    except KeyError as error:
        choices = ", ".join(_BLEND_MODES)
        raise ValueError(
            f"Unknown mask blend mode {params['mode']!r}; choose {choices}"
        ) from error


def render_layer_mask(renderer, inputs, params):
    return render_rgb_shader(
        renderer,
        "layer_mask",
        inputs,
        uniforms=_mode_uniform(params),
        image_uniforms=("tex0", "tex1", "mask"),
    )


def register_plugins(registry):
    descriptor = shader(
        "layer_mask",
        "layer/layer_mask/layer_mask.wgsl",
        derivative_contract=CONTRACT,
        interface=INTERFACE,
    )
    registry.register(
        Plugin(
            "layer_mask",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.COLOR, TextureKind.COLOR),
            params=("mode",),
            defaults={"mode": "mix"},
            shader=descriptor,
            input_uniforms=("tex0", "tex1", "mask"),
            make_uniforms=_mode_uniform,
            mode="RGB",
            render_func=render_layer_mask,
            lowering_func=lower_composite,
            method_of=(TextureKind.COLOR,),
        )
    )
    registry.register(
        Plugin(
            "mask",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.COLOR, TextureKind.COLOR),
            params=("mode",),
            defaults={"mode": "mix"},
            shader=descriptor,
            input_uniforms=("tex0", "tex1", "mask"),
            make_uniforms=_mode_uniform,
            mode="RGB",
            render_func=render_layer_mask,
            lowering_func=lower_composite,
            method_of=(TextureKind.COLOR,),
        )
    )
