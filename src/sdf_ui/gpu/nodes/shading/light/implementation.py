"""Light shader's derivative contract and native GPU program."""

from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import (
    ShaderArgument,
    ShaderDifferentiation,
)

from .reference import reference

INTERFACE = ShaderInterface(
    ImageBinding("destTex", 0, write=True), (ImageBinding("sdf", 1, read=True),),
    (Uniform("fgColor", "vec4"), Uniform("bgColor", "vec4"), Uniform("lightDir", "vec3"),
     *(Uniform(name, "float") for name in (
         "ambient", "diffuse", "specular", "shininess", "normalStrength", "bevel",
         "shadeBackground", "backgroundBevel", "heightProfile", "heightGamma",
         "height", "inflate", "inner", "outer",
     ))),
)

CONTRACT = ShaderDifferentiation(
    "light", "gpu/nodes/shading/light/light.wgsl",
    (ShaderArgument("sdf", (), "r32f"),),
    (ShaderArgument("fgColor", (4,), "vec4"), ShaderArgument("bgColor", (4,), "vec4"),
     ShaderArgument("lightDir", (3,), "vec3"),
     *(ShaderArgument(name, (), "float") for name in (
         "ambient", "diffuse", "specular", "shininess", "normalStrength", "bevel",
         "shadeBackground", "backgroundBevel", "heightProfile", "heightGamma",
         "height", "inflate", "inner", "outer",
     ))),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    zero_gradient_parameters=("heightProfile",),
    notes="Sobel stencil, normals/material/light differentiate; heightProfile is discrete.",
    sampling="clamped_sobel", reference=reference,
)
