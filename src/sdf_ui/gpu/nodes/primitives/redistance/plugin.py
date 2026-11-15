"""Recompute approximate signed distance from a sampled zero contour."""

from sdf_ui.gpu.nodes._distance import (
    read_sdf,
    write_sdf,
)
from sdf_ui.gpu.nodes._distance import redistance as compute_redistance
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind


def render_redistance(renderer, inputs, params):
    return write_sdf(renderer, compute_redistance(read_sdf(renderer, inputs)))


def register_plugins(registry):
    registry.register(
        Plugin(
            "redistance",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            render_func=render_redistance,
            method_of=(TextureKind.SDF,),
        )
    )
