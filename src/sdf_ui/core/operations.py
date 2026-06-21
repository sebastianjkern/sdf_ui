"""Runtime helpers for executing compiled shader programs."""

__docformat__ = "google"

from ..log import logger
from .context import Context


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

    dispatch_groups = getattr(ctx, "dispatch_groups", None)
    if dispatch_groups is None:
        dispatch_groups = ctx.local_size
    stats = getattr(ctx, "_active_render_stats", None)
    if stats is not None:
        stats.record_shader_dispatch(shader_name, dispatch_groups)

    shader.run(*dispatch_groups)
    logger().debug(f"Running {shader_name} shader...")
    return shader
