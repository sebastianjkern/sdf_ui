__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind


def render_shadow(renderer, inputs, params):
    from sdf_ui.core.plugins.registry import registry

    mask = registry.build(
        "generate_mask",
        inputs[0],
        inflate=params["inflate"],
        color1=(0.0, 0.0, 0.0, 0.0),
    )
    blur = registry.build("blur", mask, n=params["distance"], base=9)
    return renderer.render(registry.build("transparency", blur, params["transparency"]))


def register_plugins(registry):
    registry.register(
        Plugin(
            "shadow",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            (TextureKind.SDF,),
            params=("distance", "inflate", "transparency"),
            defaults={"distance": 10, "inflate": 0, "transparency": 0.75},
            render_func=render_shadow,
            method_of=(TextureKind.SDF,),
        )
    )
