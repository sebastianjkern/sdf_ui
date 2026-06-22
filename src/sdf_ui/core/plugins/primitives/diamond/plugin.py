__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind


def render_diamond(renderer, inputs, params):
    from sdf_ui.core.plugins.registry import registry

    center = params["center"]
    rx, ry = params["radii"]
    points = (
        (center[0], center[1] - ry),
        (center[0] + rx, center[1]),
        (center[0], center[1] + ry),
        (center[0] - rx, center[1]),
    )
    return renderer.render(registry.build("convex_polygon", points))


def register_plugins(registry):
    for name in ("diamond", "rhombus"):
        registry.register(
            Plugin(
                name,
                PluginFamily.PRIMITIVE,
                TextureKind.SDF,
                params=("center", "radii"),
                render_func=render_diamond,
                public=True,
                method_of=(TextureKind.SDF,),
            )
        )

