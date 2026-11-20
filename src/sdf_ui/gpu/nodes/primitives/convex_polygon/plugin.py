__docformat__ = "google"

from functools import reduce

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import lower_composite
from sdf_ui.gpu.nodes.common import polygon_signed_area as _polygon_area


def render_convex_polygon(renderer, inputs, params):
    from sdf_ui.gpu.nodes.registry import registry

    points = tuple(params["points"])
    if len(points) < 3:
        raise ValueError("convex_polygon requires at least 3 points")

    area = _polygon_area(points)
    if abs(area) < 1 / 1_000_000_000:
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
            lowering_func=lower_composite,
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )
