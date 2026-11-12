__docformat__ = "google"

from sdf_ui.backends.opengl.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.backends.opengl.nodes.common import lower_composite


def render_linear_gradient(renderer, inputs, params):
    import math

    from sdf_ui.backends.opengl.nodes.registry import registry

    a = params["a"]
    b = params["b"]
    ax, ay = a
    bx, by = b
    dx = ax - bx
    dy = ay - by
    line = registry.build("line", (ax + dx, ay - dy), (ax - dx, ay + dy))
    return renderer.render(
        registry.build(
            "fill",
            line,
            params["color1"],
            params["color2"],
            0,
            0,
            math.sqrt(dx * dx + dy * dy),
        )
    )


def register_plugins(registry):
    registry.register(
        Plugin(
            "linear_gradient",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            params=("a", "b", "color1", "color2"),
            render_func=render_linear_gradient,
            lowering_func=lower_composite,
            public=True,
        )
    )
