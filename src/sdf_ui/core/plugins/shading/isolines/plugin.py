__docformat__ = "google"

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.common import color, shader


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
                "spacing": 0.75,
                "line_width": 0.2,
                "feather": 0.15,
                "phase": 0.0,
            },
            shader=shader("isolines", "plugins/shading/isolines/shader.glsl"),
            input_uniforms=("sdf",),
            make_uniforms=lambda p: {
                "background": color(p["bg_color"]),
                "line_color": color(p["fg_color"]),
                "inflate": p["inflate"],
                "spacing": p["spacing"],
                "line_width": p["line_width"],
                "feather": p["feather"],
                "phase": p["phase"],
            },
            method_of=(TextureKind.SDF,),
        )
    )
