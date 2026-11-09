__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import lower_composite


def render_output(renderer, inputs, params):
    return inputs[0][params["index"]]


def register_plugins(registry):
    registry.register(
        Plugin(
            "output",
            PluginFamily.PRIMITIVE,
            TextureKind.MULTI_OUTPUT,
            (TextureKind.MULTI_OUTPUT,),
            params=("index", "kind"),
            render_func=render_output,
            lowering_func=lower_composite,
        )
    )
