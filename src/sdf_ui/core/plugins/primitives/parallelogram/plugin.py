__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind


def render_parallelogram(renderer, inputs, params):
    from sdf_ui.core.plugins.registry import registry

    center = params["center"]
    width, height = params["size"]
    skew = float(params["skew"])
    half_w = float(width) * 0.5
    half_h = float(height) * 0.5
    points = (
        (center[0] - half_w + skew, center[1] - half_h),
        (center[0] + half_w + skew, center[1] - half_h),
        (center[0] + half_w - skew, center[1] + half_h),
        (center[0] - half_w - skew, center[1] + half_h),
    )
    return renderer.render(registry.build("convex_polygon", points))


def register_plugins(registry):
    registry.register(
        Plugin(
            "parallelogram",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("center", "size", "skew"),
            defaults={"skew": 0.0},
            render_func=render_parallelogram,
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )

