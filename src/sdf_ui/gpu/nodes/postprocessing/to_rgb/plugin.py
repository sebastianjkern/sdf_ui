__docformat__ = "google"

from functools import partial

from sdf_ui.gpu.nodes.base import PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import register_single_texture_shader

from .implementation import CONTRACT, INTERFACE

register_plugins = partial(
    register_single_texture_shader,
    name="to_rgb",
    family=PluginFamily.POSTPROCESSING,
    texture_kind=TextureKind.COLOR,
    shader_path="postprocessing/to_rgb/to_rgb.wgsl",
    contract=CONTRACT,
    interface=INTERFACE,
    mode="RGB",
)
