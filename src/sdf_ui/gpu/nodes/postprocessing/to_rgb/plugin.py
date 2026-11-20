__docformat__ = "google"

from functools import partial

from sdf_ui.gpu.nodes.base import PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import register_single_texture_shader

from .implementation import CONTRACT, INTERFACE, PROGRAM

register_plugins = partial(
    register_single_texture_shader,
    name="to_rgb",
    family=PluginFamily.POSTPROCESSING,
    texture_kind=TextureKind.COLOR,
    shader_path="postprocessing/to_rgb/shader.glsl",
    contract=CONTRACT,
    program=PROGRAM,
    interface=INTERFACE,
    mode="RGB",
)
