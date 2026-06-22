__docformat__ = "google"

from functools import reduce

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind


def _polygon_area(points):
    area = 0.0
    for index, point in enumerate(points):
        x0, y0 = point
        x1, y1 = points[(index + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return area * 0.5


def render_convex_polygon(renderer, inputs, params):
    from sdf_ui.core.plugins.registry import registry

    points = tuple(params["points"])
    if len(points) < 3:
        raise ValueError("convex_polygon requires at least 3 points")

    area = _polygon_area(points)
    if abs(area) < 1e-9:
        raise ValueError("convex_polygon points must not be collinear")

    winding = 1.0 if area > 0 else -1.0
    half_planes = []
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        edge_x = next_point[0] - point[0]
        edge_y = next_point[1] - point[1]
        outward = (winding * edge_y, winding * -edge_x)
        half_planes.append(registry.build("half_plane", point, outward))

    shape = reduce(lambda a, b: a.intersection(b), half_planes)
    return renderer.render(shape)


def register_plugins(registry):
    registry.register(
        Plugin(
            "convex_polygon",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("points",),
            render_func=render_convex_polygon,
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )

