__docformat__ = "google"

from functools import partial

from sdf_ui.gpu.nodes.base import PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import register_single_texture_shader

from .implementation import CONTRACT, INTERFACE, PROGRAM

register_plugins = partial(
    register_single_texture_shader,
    name="union",
    family=PluginFamily.PRIMITIVE,
    texture_kind=TextureKind.SDF,
    shader_path="primitives/union/shader.glsl",
    contract=CONTRACT,
    program=PROGRAM,
    interface=INTERFACE,
)
