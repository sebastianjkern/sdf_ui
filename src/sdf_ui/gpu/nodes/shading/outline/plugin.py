__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import color, shader

from .implementation import CONTRACT, INTERFACE


def register_plugins(registry):
    descriptor = shader(
        "outline",
        "shading/outline/outline.wgsl",
        derivative_contract=CONTRACT,
        interface=INTERFACE,
    )
    for name in ("outline", "stroke"):
        registry.register(
            Plugin(
                name,
                PluginFamily.SHADING,
                TextureKind.COLOR,
                (TextureKind.SDF,),
                params=("fg_color", "bg_color", "inflate", "width"),
                defaults={
                    "bg_color": (0.0, 0.0, 0.0, 0.0),
                    "inflate": 0.0,
                    "width": 1.5,
                },
                shader=descriptor,
                input_uniforms=("sdf",),
                make_uniforms=lambda p: {
                    "background": color(p["bg_color"]),
                    "outline": color(p["fg_color"]),
                    "inflate": p["inflate"],
                    "width": p["width"],
                },
                method_of=(TextureKind.SDF,),
            )
        )
