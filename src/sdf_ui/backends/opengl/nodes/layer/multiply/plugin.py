__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.common import lower_composite, render_rgb_shader, shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def render_multiply(renderer, inputs, params):
    return render_rgb_shader(
        renderer, "multiply", inputs, image_uniforms=("mask1", "mask2")
    )


def register_plugins(registry):
    registry.register(
        Plugin(
            "multiply",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.COLOR),
            shader=shader("multiply", "layer/multiply/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            input_uniforms=("mask1", "mask2"),
            mode="RGB",
            render_func=render_multiply,
            lowering_func=lower_composite,
            method_of=(TextureKind.COLOR,),
        )
    )
