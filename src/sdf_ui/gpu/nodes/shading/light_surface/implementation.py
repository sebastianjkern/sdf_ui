from sdf_ui.gpu.nodes.base import ImageBinding, ShaderInterface, Uniform
from sdf_ui.gpu.nodes.differentiation import ShaderArgument, ShaderDifferentiation

from .reference import reference

INTERFACE = ShaderInterface(
    output=ImageBinding("destTex", 0, write=True),
    inputs=(
        ImageBinding("sdf", 1, read=True),
        ImageBinding("heightField", 2, read=True),
    ),
    uniforms=(
        Uniform("fgColor", "vec4"),
        Uniform("lightDir", "vec3"),
        *(
            Uniform(name, "float")
            for name in (
                "ambient",
                "diffuse",
                "specular",
                "shininess",
            )
        ),
    ),
)

CONTRACT = ShaderDifferentiation(
    "light_surface",
    "gpu/nodes/shading/light_surface/light_surface.wgsl",
    (
        ShaderArgument("sdf", (), "r32f"),
        ShaderArgument("heightField", (), "r32f", port_kind="height"),
    ),
    (
        ShaderArgument("fgColor", (4,), "vec4"),
        ShaderArgument("lightDir", (3,), "vec3"),
        *(
            ShaderArgument(name, (), "float")
            for name in (
                "ambient",
                "diffuse",
                "specular",
                "shininess",
            )
        ),
    ),
    (ShaderArgument("destTex", (4,), "rgba8"),),
    notes="Clamped Sobel normals from the height field; alpha is centered one-pixel SDF coverage.",
    sampling="clamped_sobel",
    reference=reference,
)
