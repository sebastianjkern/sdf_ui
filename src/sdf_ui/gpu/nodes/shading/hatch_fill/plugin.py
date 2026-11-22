__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import color, shader

from .implementation import CONTRACT, INTERFACE


def register_plugins(registry):
    registry.register(
        Plugin(
            "hatch_fill",
            PluginFamily.SHADING,
            TextureKind.COLOR,
            (TextureKind.SDF,),
            params=(
                "fg_color",
                "inflate",
                "angle",
                "spacing",
                "phase",
                "width",
                "origin_x",
                "origin_y",
            ),
            defaults={
                "inflate": 0.0,
                "angle": 0.0,
                "spacing": 8.0,
                "phase": 0.0,
                "width": 1.0,
                "origin_x": 0.0,
                "origin_y": 0.0,
            },
            shader=shader(
                "hatch_fill",
                "shading/hatch_fill/hatch_fill.wgsl",
                derivative_contract=CONTRACT,
                interface=INTERFACE,
            ),
            input_uniforms=("sdf",),
            make_uniforms=lambda p: {
                "fg_color": color(p["fg_color"]),
                "inflate": p["inflate"],
                "angle": p["angle"],
                "spacing": p["spacing"],
                "phase": p["phase"],
                "width": p["width"],
                "origin_x": p["origin_x"],
                "origin_y": p["origin_y"],
            },
            public=False,
        )
    )
