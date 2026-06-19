__docformat__ = "google"

from .context import Context
from .sdf import SDFTexture
from .shaders import Shaders
from ..log import logger


def run_shader(ctx: Context, shader_name: str, *, uniforms=None, image_bindings=None):
    """Bind uniforms/images and execute a compute shader for the context size.

    image_bindings maps texture objects to ModernGL image binding options. Each
    value is a tuple of (texture, location, read, write).
    """
    shader = ctx.get_shader(shader_name)

    for name, value in (uniforms or {}).items():
        shader[name] = value

    for texture, location, read, write in (image_bindings or ()):
        texture.bind_to_image(location, read=read, write=write)

    shader.run(*ctx.local_size)
    logger().debug(f"Running {shader_name} shader...")
    return shader


def interpolate(ctx: Context, tex0: SDFTexture, tex1: SDFTexture, t=0.5) -> SDFTexture:
    """
    Interpolates between two signed distance field (SDF) textures.

    Args:
    - tex0: The first SDFTexture to interpolate from.
    - tex1: The second SDFTexture to interpolate to.
    - t (float): The interpolation factor. It should be in the range [0, 1],
                 where 0 corresponds to tex0, and 1 corresponds to tex1.
                 Default is 0.5, resulting in a mid-point interpolation.

    Returns:
    A new SDFTexture representing the interpolation between tex0 and tex1.

    Example:
    >>> sdf_texture_0 = SDFTexture(...)
    >>> sdf_texture_1 = SDFTexture(...)
    >>> interpolated_texture = interpolate(sdf_texture_0, sdf_texture_1, t=0.3)
    """
    tex = ctx.r32f()

    run_shader(
        ctx,
        Shaders.INTERPOLATION,
        uniforms={
            "destTex": 0,
            "sdf0": 1,
            "sdf1": 2,
            "t": t,
        },
        image_bindings=(
            (tex, 0, False, True),
            (tex0.tex, 1, True, False),
            (tex1.tex, 2, True, False),
        ),
    )

    return SDFTexture(tex, context=ctx)
