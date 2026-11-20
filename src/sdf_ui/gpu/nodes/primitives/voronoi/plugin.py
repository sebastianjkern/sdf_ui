"""Voronoi cell distance-field factory."""

import numpy as np

from sdf_ui.core.sdf import SDFTexture
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind


def _validated_points(params):
    points = np.asarray(params["points"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("voronoi requires at least two 2D points")
    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
        raise ValueError("voronoi points must be finite and distinct")
    return points


def _cell_index(params, point_count):
    cell = params["cell"]
    if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < point_count:
        raise ValueError(f"cell must be an integer from 0 to {point_count - 1}")
    return cell


def _cell_field(points, cell, width, height):
    yy, xx = np.mgrid[0:height, 0:width]
    grid = np.stack((xx, yy), axis=-1).astype(np.float32)
    origin = points[cell]
    result = np.full((height, width), -np.inf, dtype=np.float32)
    for index, other in enumerate(points):
        if index != cell:
            result = _bisector_field(result, grid, origin, other)
    return result


def _bisector_field(current, grid, origin, other):
    delta = other - origin
    length = float(np.linalg.norm(delta))
    if length <= 1 / 1_000_000:
        raise ValueError("voronoi points must be distinct")
    midpoint = (origin + other) * 0.5
    signed = np.sum((grid - midpoint) * (delta / length), axis=-1)
    return np.maximum(current, signed)


def render_voronoi(renderer, inputs, params):
    """Render the signed distance to one unbounded Voronoi cell."""
    points = _validated_points(params)
    cell = _cell_index(params, len(points))
    width, height = renderer.ctx.size
    values = _cell_field(points, cell, width, height)
    texture = renderer.ctx.r32f()
    texture.write(values.astype(np.float32).tobytes())
    return SDFTexture(tex=texture, context=renderer.ctx)


def register_plugins(registry):
    registry.register(
        Plugin(
            "voronoi",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("points", "cell"),
            defaults={"cell": 0},
            render_func=render_voronoi,
            public=True,
        )
    )
