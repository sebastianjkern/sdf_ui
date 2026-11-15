"""Image and uniform layout for the local distance correction shader."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform

INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True),
    (ImageBinding("sdf", 1, read=True),),
    (Uniform("epsilon", "float"), Uniform("band", "float")),
)
