__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import color, shader


def _uniforms(params):
    return {
        "background": color(params["bg_color"]),
        "line_color": color(params["fg_color"]),
        "inflate": params["inflate"],
        "spacing": params["spacing"],
        "line_width": params["line_width"],
        "feather": params["feather"],
        "phase": params["phase"],
    }


def register_plugins(registry):
    registry.register(
        Plugin(
            "isolines",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            (TextureKind.SDF,),
            params=(
                "fg_color",
                "bg_color",
                "inflate",
                "spacing",
                "line_width",
                "feather",
                "phase",
            ),
            defaults={
                "bg_color": (0.0, 0.0, 0.0, 0.0),
                "inflate": 0.0,
                "spacing": 12.0,
                "line_width": 1.0,
                "feather": 0.0,
                "phase": 0.0,
            },
            shader=shader("isolines", "plugins/shading/isolines/shader.glsl"),
            input_uniforms=("sdf",),
            make_uniforms=_uniforms,
            method_of=(TextureKind.SDF,),
        )
    )

