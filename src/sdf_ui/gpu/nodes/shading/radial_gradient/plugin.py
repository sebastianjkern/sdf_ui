__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import lower_composite


def render_radial_gradient(renderer, inputs, params):
    from sdf_ui.gpu.nodes.registry import registry

    disc = registry.build("circle", params["a"], 0)
    return renderer.render(
        registry.build(
            "fill",
            disc,
            params["color1"],
            params["color2"],
            0,
            params["inner"],
            params["outer"],
        )
    )


def register_plugins(registry):
    registry.register(
        Plugin(
            "radial_gradient",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            params=("a", "color1", "color2", "inner", "outer"),
            defaults={"inner": 0, "outer": 100},
            render_func=render_radial_gradient,
            lowering_func=lower_composite,
            public=True,
        )
    )
