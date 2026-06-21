__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind


def render_generate_mask(renderer, inputs, params):
    from sdf_ui.core.plugins.registry import registry

    return renderer.render(registry.build("fill", inputs[0], params["color1"], params["color0"], params["inflate"]).to_rgb())


def register_plugins(registry):
    registry.register(Plugin("generate_mask", PluginFamily.SHADING, TextureKind.COLOR, (TextureKind.SDF,), params=("inflate", "color0", "color1"), defaults={"inflate": 0.0, "color0": (0.0, 0.0, 0.0, 1.0), "color1": (1.0, 1.0, 1.0, 1.0)}, render_func=render_generate_mask, method_of=(TextureKind.SDF,)))
