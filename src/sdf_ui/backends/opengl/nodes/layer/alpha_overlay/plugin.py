__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.common import lower_composite, render_rgb_shader, shader
from .implementation import CONTRACT, INTERFACE, PROGRAM


def render_alpha_overlay(renderer, inputs, params):
    return render_rgb_shader(
        renderer, "overlay", inputs, image_uniforms=("tex1", "tex0")
    )


def register_plugins(registry):
    registry.register(
        Plugin(
            "alpha_overlay",
            PluginFamily.LAYER,
            TextureKind.COLOR,
            (TextureKind.COLOR, TextureKind.COLOR),
            shader=shader("overlay", "layer/alpha_overlay/shader.glsl", derivative_contract=CONTRACT, program=PROGRAM, interface=INTERFACE),
            input_uniforms=("tex1", "tex0"),
            mode="RGB",
            render_func=render_alpha_overlay,
            lowering_func=lower_composite,
            method_of=(TextureKind.COLOR,),
        )
    )
