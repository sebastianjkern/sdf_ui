__docformat__ = "google"

from .context import Context
from ..log import logger


def run_shader(ctx: Context, shader_name: str, *, uniforms=None, image_bindings=None):
    """Bind uniforms/images and execute a compute shader for the context size.

    image_bindings maps texture objects to ModernGL image binding options. Each
    value is a tuple of (texture, location, read, write).
    """
    shader = ctx.get_shader(shader_name)

    for name, value in (uniforms or {}).items():
        shader[name] = value

    for texture, location, read, write in image_bindings or ():
        texture.bind_to_image(location, read=read, write=write)

    shader.run(*ctx.local_size)
    logger().debug(f"Running {shader_name} shader...")
    return shader
