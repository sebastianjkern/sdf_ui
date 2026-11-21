__docformat__ = "google"

from functools import partial

from sdf_ui.gpu.nodes.base import PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import register_single_texture_shader

from .implementation import CONTRACT, INTERFACE

register_plugins = partial(
    register_single_texture_shader,
    name="to_lab",
    family=PluginFamily.POSTPROCESSING,
    texture_kind=TextureKind.COLOR,
    shader_path="postprocessing/to_lab/to_lab.wgsl",
    contract=CONTRACT,
    interface=INTERFACE,
    mode="LAB",
)
