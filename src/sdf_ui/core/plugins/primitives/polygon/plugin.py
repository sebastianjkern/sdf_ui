__docformat__ = "google"

import numpy as np

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind


def _polygon_area(points):
    area = 0.0
    for index, point in enumerate(points):
        x0, y0 = point
        x1, y1 = points[(index + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return area * 0.5


def _point_segment_distances(points_grid, a, b):
    ba = np.array(b, dtype=np.float32) - np.array(a, dtype=np.float32)
    pa = points_grid - np.array(a, dtype=np.float32)
    denom = float(np.dot(ba, ba))
    if denom <= 1e-12:
        return np.sum(pa * pa, axis=-1)

    h = np.clip(np.sum(pa * ba, axis=-1) / denom, 0.0, 1.0)
    delta = pa - h[..., None] * ba
    return np.sum(delta * delta, axis=-1)


def _point_in_polygon(points_grid, points):
    x = points_grid[..., 0]
    y = points_grid[..., 1]
    inside = np.zeros(points_grid.shape[:2], dtype=bool)

    for index, point in enumerate(points):
        x0, y0 = point
        x1, y1 = points[(index + 1) % len(points)]
        crosses = ((y0 > y) != (y1 > y)) & (
            x < ((x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0)
        )
        inside ^= crosses

    return inside


def render_polygon(renderer, inputs, params):
    from sdf_ui.core.sdf import SDFTexture

    ctx = renderer.ctx
    points = tuple(tuple(point) for point in params["points"])
    if len(points) < 3:
        raise ValueError("polygon requires at least 3 points")

    if abs(_polygon_area(points)) < 1e-9:
        raise ValueError("polygon points must enclose a non-zero area")

    width, height = ctx.size
    yy, xx = np.mgrid[0:height, 0:width]
    grid = np.stack((xx.astype(np.float32), yy.astype(np.float32)), axis=-1)
    distance_sq = np.full((height, width), np.inf, dtype=np.float32)

    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        distance_sq = np.minimum(
            distance_sq, _point_segment_distances(grid, point, next_point)
        )

    field = np.sqrt(distance_sq).astype(np.float32)
    field[_point_in_polygon(grid, points)] *= -1.0

    tex = ctx.r32f()
    tex.write(field.tobytes())
    return SDFTexture(tex=tex, context=ctx)


def register_plugins(registry):
    registry.register(
        Plugin(
            "polygon",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("points",),
            render_func=render_polygon,
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )

