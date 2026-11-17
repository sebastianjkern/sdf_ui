"""Finite SDF grid repetition."""

__docformat__ = "google"

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.common import shader

from .implementation import CONTRACT, INTERFACE, PROGRAM


def _uniforms(params):
    import math

    import numpy as np

    columns, rows = params["columns"], params["rows"]
    if isinstance(columns, bool) or int(columns) != columns or columns < 1:
        raise ValueError("columns must be a positive integer")
    if isinstance(rows, bool) or int(rows) != rows or rows < 1:
        raise ValueError("rows must be a positive integer")
    spacing = np.asarray(params["spacing"], dtype=np.float32)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing must contain two finite positive values")
    center = params["center"]
    if len(center) != 2 or not all(math.isfinite(float(value)) for value in center):
        raise ValueError("center must contain two finite values")
    return {
        "columns": float(columns),
        "rows": float(rows),
        "spacing": spacing,
        "center": center,
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
