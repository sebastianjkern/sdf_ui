__docformat__ = "google"

import math

from .context import Context
from .operations import run_shader
from .sdf import SDFTexture
from .shaders import Shaders


def _render_sdf(ctx: Context, shader_name: str, uniforms) -> SDFTexture:
    tex = ctx.r32f()
    run_shader(
        ctx,
        shader_name,
        uniforms={"destTex": 0, **uniforms},
        image_bindings=((tex, 0, False, True),),
    )
    return SDFTexture(tex, context=ctx)


def rounded_rect(ctx, center, size, corner_radius, angle=0/360*math.pi) -> SDFTexture:
    """
    Creates a signed distance field (SDF) texture representing a rounded rectangle.

    Args:
    - center: The center coordinates of the rectangle.
    - size: The size of the rectangle (width, height).
    - corner_radius: The radius of the rounded corners.
    - angle (float): The rotation angle of the rectangle in radians. Default is 0.

    Returns:
    A new SDFTexture representing a rounded rectangle.

    Example:
    >>> center = (0.0, 0.0)
    >>> size = (2.0, 1.0)
    >>> corner_radius = 0.3
    >>> angle = 45/360*math.pi
    >>> rounded_rect_texture = rounded_rect(center, size, corner_radius, angle)
    """

    return _render_sdf(
        ctx,
        Shaders.RECT,
        {
            "offset": center,
            "size": size,
            "corner_radius": corner_radius,
            "angle": angle,
        },
    )


def disc(ctx, center, radius) -> SDFTexture:
    """
    Creates a signed distance field (SDF) texture representing a filled disc (circle).

    Args:
    - center: The center coordinates of the disc.
    - radius: The radius of the disc.

    Returns:
    A new SDFTexture representing a filled disc.

    Example:
    >>> center = (0.0, 0.0)
    >>> radius = 1.5
    >>> disc_texture = disc(center, radius)
    """

    return _render_sdf(
        ctx,
        Shaders.CIRCLE,
        {
            "offset": center,
            "radius": radius,
        },
    )


def triangle(ctx, point_1, point_2, point_3) -> SDFTexture:
    """

    Renders a triangle using a signed distance field (SDF) approach.

    Args:
        ctx (SDFContext): The context object for the signed distance field rendering.
        point_1 (tuple): Coordinates of the first vertex of the triangle (x, y).
        point_2 (tuple): Coordinates of the second vertex of the triangle (x, y).
        point_3 (tuple): Coordinates of the third vertex of the triangle (x, y).

    Returns:
        SDFTexture: An SDFTexture object representing the rendered triangle.

    Note:
        The triangle is rendered using the specified vertices and the shader associated with
        Shaders.TRIANGLE. The resulting SDFTexture is returned for further use or analysis.

    Example:
        >>> point1 = (0.0, 0.0)
        >>> point2 = (1.0, 0.0)
        >>> point3 = (0.5, 1.0)
        >>> sdf_texture = triangle(ctx, point1, point2, point3)
    """

    return _render_sdf(
        ctx,
        Shaders.TRIANGLE,
        {
            "point0": point_1,
            "point1": point_2,
            "point2": point_3,
        },
    )

def bezier(ctx, a, b, c) -> SDFTexture:
    """
    Creates a signed distance field (SDF) texture representing a quadratic Bezier curve.

    Args:
    - a: The starting point of the Bezier curve.
    - b: The control point influencing the curvature of the curve.
    - c: The ending point of the Bezier curve.

    Returns:
    A new SDFTexture representing a quadratic Bezier curve.

    Example:
    >>> a = (0.0, 0.0)
    >>> b = (1.0, 2.0)
    >>> c = (2.0, 0.0)
    >>> bezier_texture = bezier(a, b, c)
    """

    return _render_sdf(
        ctx,
        Shaders.BEZIER,
        {
            "a": a,
            "b": b,
            "c": c,
        },
    )


def line(ctx, a, b) -> SDFTexture:
    """
    Creates a signed distance field (SDF) texture representing a straight line segment.

    Args:
    - a: The starting point of the line segment.
    - b: The ending point of the line segment.

    Returns:
    A new SDFTexture representing a straight line.

    Example:
    >>> a = (0.0, 0.0)
    >>> b = (2.0, 1.0)
    >>> line_texture = line(a, b)
    """

    return _render_sdf(
        ctx,
        Shaders.LINE,
        {
            "a": a,
            "b": b,
        },
    )


def grid(ctx, offset, size, angle=0/360*math.pi) -> SDFTexture:
    """
    Creates a signed distance field (SDF) texture representing a grid.

    Args:
    - offset: The center coordinates of the grid.
    - size: The size of each cell in the grid (width, height).
    - angle (float): The rotation angle of the grid in radians. Default is 0.

    Returns:
    A new SDFTexture representing a grid.

    Example:
    >>> offset = (0.0, 0.0)
    >>> size = (1.0, 1.0)
    >>> angle = 45/360*math.pi
    >>> grid_texture = grid(offset, size, angle)
    """

    return _render_sdf(
        ctx,
        Shaders.GRID,
        {
            "grid_size": size,
            "offset": offset,
            "angle": angle,
        },
    )


