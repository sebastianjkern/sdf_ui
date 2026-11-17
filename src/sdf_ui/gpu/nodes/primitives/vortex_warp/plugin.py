"""Displace an SDF by a smooth sum of vortex vector fields."""

import numpy as np

from sdf_ui.core.field import FieldNamespace
from sdf_ui.gpu.nodes._vortex import normalize_vortices, validate_radius
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind


def render_vortex_warp(renderer, inputs, params):
    vortices = normalize_vortices(params["vortices"])
    radius = validate_radius(params["radius"])
    warp_strength = float(params["strength"])
    if not np.isfinite(warp_strength):
        raise ValueError("warp strength must be finite")
    namespace = FieldNamespace()
    combined = None
    for cx, cy, vortex_strength in vortices:
        vortex = namespace.vortex((float(cx), float(cy)), radius, strength=float(vortex_strength))
        combined = vortex if combined is None else combined.add(vortex)
    return renderer.render(inputs[0].warp(combined, strength=warp_strength * radius))


def register_plugins(registry):
    registry.register(
        Plugin(
            "vortex_warp",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            params=("vortices", "radius", "strength"),
            defaults={"strength": 0.8},
            render_func=render_vortex_warp,
            method_of=(TextureKind.SDF,),
        )
    )
