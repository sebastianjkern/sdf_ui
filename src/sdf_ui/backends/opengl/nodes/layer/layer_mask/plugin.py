__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.common import lower_composite, render_rgb_shader, shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def render_layer_mask(renderer, inputs, params):
    return render_rgb_shader(
        renderer, "layer_mask", inputs, image_uniforms=("tex0", "tex1", "mask")
    )


def register_plugins(registry):
    descriptor = shader("layer_mask", "layer/layer_mask/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE)
    registry.register(
        Plugin(
            "layer_mask",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.COLOR, TextureKind.COLOR),
            shader=descriptor,
            input_uniforms=("tex0", "tex1", "mask"),
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
            shader=descriptor,
            input_uniforms=("tex0", "tex1", "mask"),
            mode="RGB",
            render_func=render_layer_mask,
            lowering_func=lower_composite,
            method_of=(TextureKind.COLOR,),
        )
    )
