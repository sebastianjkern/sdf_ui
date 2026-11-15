"""Apply a first-order unit-gradient correction to a sampled SDF."""

from sdf_ui.gpu.nodes._distance import normalize_distance, read_sdf, write_sdf
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind


def render_normalize_distance(renderer, inputs, params):
    field = read_sdf(renderer, inputs)
    return write_sdf(
        renderer,
        normalize_distance(field, params["epsilon"], params["band"]),
    )


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
