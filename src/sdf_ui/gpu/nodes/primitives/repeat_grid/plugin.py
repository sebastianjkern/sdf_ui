"""Finite SDF grid repetition."""

__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import shader

from .implementation import CONTRACT, INTERFACE, PROGRAM


def _positive_integer(params, name):
    value = params[name]
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return float(value)


def _spacing(params):
    import numpy as np

    spacing = np.asarray(params["spacing"], dtype=np.float32)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing must contain two finite positive values")
    return spacing


def _center(params):
    import math

    center = params["center"]
    if len(center) != 2 or not all(math.isfinite(float(value)) for value in center):
        raise ValueError("center must contain two finite values")
    return center


def _uniforms(params):
    return {
        "columns": _positive_integer(params, "columns"),
        "rows": _positive_integer(params, "rows"),
        "spacing": _spacing(params),
        "center": _center(params),
    }


def register_plugins(registry):
    registry.register(
        Plugin(
            "repeat_grid",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            (TextureKind.SDF,),
            params=("columns", "rows", "spacing", "center"),
            defaults={},
            shader=shader(
                "repeat_grid",
                "primitives/repeat_grid/shader.glsl",
                derivative_contract=CONTRACT,
                program=PROGRAM,
                interface=INTERFACE,
            ),
            make_uniforms=_uniforms,
            input_uniforms=("sdf0",),
            lowering_func=None,
            method_of=(TextureKind.SDF,),
        )
    )
