__docformat__ = "google"

from math import cos, pi, sin

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind


def render_ngon(renderer, inputs, params):
    from sdf_ui.core.plugins.registry import registry

    sides = int(params["sides"])
    if sides < 3:
        raise ValueError("ngon requires at least 3 sides")

    center = params["center"]
    radius = float(params["radius"])
    rotation = float(params["rotation"])

    points = tuple(
        (
            center[0] + radius * cos(rotation + 2.0 * pi * index / sides),
            center[1] + radius * sin(rotation + 2.0 * pi * index / sides),
        )
        for index in range(sides)
    )
    return renderer.render(registry.build("convex_polygon", points))


def register_plugins(registry):
    registry.register(
        Plugin(
            "ngon",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", "radius", "sides", "rotation"),
            defaults={"sides": 6, "rotation": 0.0},
            render_func=render_ngon,
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )

