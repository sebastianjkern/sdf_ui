__docformat__ = "google"

import math

from .context import Context
from .color import ColorSpaceMode, ColorTexture
from .operations import run_shader
from .primitives import disc, line
from .shaders import Shaders


def _render_color(ctx: Context, shader_name: str, mode: str, uniforms=None) -> ColorTexture:
    tex = ctx.rgba8()
    run_shader(
        ctx,
        shader_name,
        uniforms={"destTex": 0, **(uniforms or {})},
        image_bindings=((tex, 0, False, True),),
    )
    return ColorTexture(tex, context=ctx, mode=mode)


def clear_color(ctx: Context, color) -> ColorTexture:
    """
    Creates a color texture filled with a specified color.

    Args:
    - color: The color to fill the texture with. Should be in RGBA format.

    Returns:
    A new ColorTexture filled with the specified color.

    Example:
    >>> color = (1.0, 0.0, 0.0, 1.0)  # Red color
    >>> clear_color_texture = clear_color(color)
    """
    return _render_color(ctx, Shaders.CLEAR_COLOR, ColorSpaceMode.LAB, {"color": color})


def perlin_noise(ctx: Context) -> ColorTexture:
    """
    Generates Perlin noise and returns it as a color texture.

    Returns:
    A new ColorTexture representing Perlin noise.

    Example:
    >>> perlin_texture = perlin_noise()
    """
    return _render_color(ctx, Shaders.PERLIN_NOISE, ColorSpaceMode.RGB)


def film_grain(ctx: Context) -> ColorTexture:
    """
    Generates a film grain effect and returns it as a color texture.

    Returns:
    A new ColorTexture representing a film grain effect.

    Example:
    >>> film_grain_texture = film_grain()
    """
    return _render_color(ctx, Shaders.FILM_GRAIN, ColorSpaceMode.RGB)


def linear_gradient(context: Context, a, b, color1, color2):
    """
    Generates a linear gradient between two points and returns it as a ColorTexture.

    Args:
    - a: The starting point of the gradient.
    - b: The ending point of the gradient.
    - color1: The color at the starting point of the gradient.
    - color2: The color at the ending point of the gradient.

    Returns:
    A ColorTexture representing a linear gradient between color1 and color2.

    Example:
    >>> start_point = (0.0, 0.0)
    >>> end_point = (1.0, 1.0)
    >>> color_at_start = (1.0, 0.0, 0.0, 1.0)  # Red
    >>> color_at_end = (0.0, 0.0, 1.0, 1.0)    # Blue
    >>> gradient_texture = linear_gradient(start_point, end_point, color_at_start, color_at_end)
    """

    ax, ay = a
    bx, by = b

    dx = ax - bx
    dy = ay - by

    cx = ax + dx
    cy = ax - dy

    ex = ax - dx
    ey = ay + dy

    return line(context, (cx, cy), (ex, ey)) \
        .fill(color1, color2, 0, inner=0, outer=math.sqrt(dx * dx + dy * dy))


def radial_gradient(context: Context, a, color1, color2, inner=0, outer=100):
    """
    Generates a radial gradient centered at a point and returns it as a ColorTexture.

    Args:
    - context: The sdf_ui rendering context
    - a: The center point of the radial gradient.
    - color1: The color at the center of the gradient.
    - color2: The color at the outer edge of the gradient.
    - inner (float): The inner radius of the gradient. Default is 0.
    - outer (float): The outer radius of the gradient. Default is 100.

    Returns:
    A ColorTexture representing a radial gradient between color1 and color2.

    Example:
    >>> center_point = (0.0, 0.0)
    >>> color_at_center = (1.0, 0.0, 0.0, 1.0)  # Red
    >>> color_at_outer = (0.0, 0.0, 1.0, 1.0)    # Blue
    >>> gradient_texture = radial_gradient(center_point, color_at_center, color_at_outer)
    """
    return disc(context, a, 0).fill(color1, color2, 0, inner=inner, outer=outer)
