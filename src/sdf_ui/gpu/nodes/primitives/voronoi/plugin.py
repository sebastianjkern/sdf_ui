"""Voronoi cell distance-field factory."""

import numpy as np

from sdf_ui.core.sdf import SDFTexture
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind


def render_voronoi(renderer, inputs, params):
    """Render the signed distance to one unbounded Voronoi cell."""
    points = np.asarray(params["points"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("voronoi requires at least two 2D points")
    if not np.isfinite(points).all():
        raise ValueError("voronoi points must be finite")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("voronoi points must be distinct")
    cell = params["cell"]
    if (
        isinstance(cell, bool)
        or not isinstance(cell, int)
        or not 0 <= cell < len(points)
    ):
        raise ValueError(f"cell must be an integer from 0 to {len(points) - 1}")

    width, height = renderer.ctx.size
    yy, xx = np.mgrid[0:height, 0:width]
    grid = np.stack((xx, yy), axis=-1).astype(np.float32)
    origin = points[cell]
    # Each competitor contributes the signed distance to its perpendicular
    # bisector with this site's point. The cell is their intersection.
    field = np.full((height, width), -np.inf, dtype=np.float32)
    for index, other in enumerate(points):
        if index == cell:
            continue
        delta = other - origin
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            raise ValueError("voronoi points must be distinct")
        midpoint = (origin + other) * 0.5
        signed = np.sum((grid - midpoint) * (delta / length), axis=-1)
        field = np.maximum(field, signed)

    texture = renderer.ctx.r32f()
    texture.write(field.astype(np.float32).tobytes())
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
