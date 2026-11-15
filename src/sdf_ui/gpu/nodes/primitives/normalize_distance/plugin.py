"""Apply a first-order unit-gradient correction to a sampled SDF on GPU."""

from sdf_ui.core.sdf import SDFTexture
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, ShaderFile, TextureKind

from .implementation import INTERFACE

# The shader interface binds the WGSL node kernel. This operation remains
# outside graph differentiation, as it did when it used a host render function.
DESCRIPTOR = ShaderFile("normalize_distance", "", interface=INTERFACE)


def render_normalize_distance(renderer, inputs, params):
    output = renderer.ctx.r32f()
    try:
        renderer.ctx.run_node_shader(
            DESCRIPTOR,
            output,
            {"sdf": inputs[0].tex},
            {"epsilon": params["epsilon"], "band": params["band"]},
        )
    except BaseException:
        output.close()
        raise
    return SDFTexture(tex=output, context=renderer.ctx)


def register_plugins(registry):
    registry.register(
        Plugin(
            "normalize_distance",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            params=("epsilon", "band"),
            defaults={"epsilon": 1e-3, "band": 32.0},
            render_func=render_normalize_distance,
            method_of=(TextureKind.SDF,),
        )
    )
