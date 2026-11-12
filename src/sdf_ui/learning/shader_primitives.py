"""Analytic autodiff rules for standalone primitive compute shaders.

``RULES[name](context, inputs, params)`` uses registry shader names and raw
GLSL uniform names. Context exposes Tensor pixel grids ``x``, ``y`` and
``size=(width, height)``; input images are scalar ``(height, width)`` Tensors.
The masked union returns ``(distance, rgba_mask)`` before rgba8 quantization.

Discrete angle masks, integer addresses and closest-point selection are frozen
for each evaluation. Gradients are exact within each smooth branch. Norms and
absolute values use zero at the origin; hard extrema split ties equally among
their active candidates. Repeat spacing and transform kind have zero derivatives.
Bézier roots are numerical branch decisions, with analytic envelope gradients
through the selected curve points, rather than numerical differentiation.
"""

from ..backends.opengl.nodes.primitives.circle.implementation import reference as _circle
from ..backends.opengl.nodes.primitives.abs.implementation import reference as _abs
from ..backends.opengl.nodes.primitives.union.implementation import reference as _union
from ..backends.opengl.nodes.primitives.intersection.implementation import reference as _intersection
from ..backends.opengl.nodes.primitives.subtract.implementation import reference as _subtract
from ..backends.opengl.nodes.primitives.interpolate.implementation import reference as _interpolate
from ..backends.opengl.nodes.primitives.ring.implementation import reference as _ring
from ..backends.opengl.nodes.primitives.line.implementation import reference as _line
from ..backends.opengl.nodes.primitives.capsule.implementation import reference as _capsule
from ..backends.opengl.nodes.primitives.half_plane.implementation import reference as _half_plane
from ..backends.opengl.nodes.primitives.ellipse.implementation import reference as _ellipse
from ..backends.opengl.nodes.primitives.smooth_union.implementation import reference as _smooth_min
from ..backends.opengl.nodes.primitives.grid.implementation import reference as _grid
from ..backends.opengl.nodes.primitives.rect.implementation import reference as _rect
from ..backends.opengl.nodes.primitives.masked_union.implementation import reference as _masked_union
from ..backends.opengl.nodes.primitives.arc.implementation import reference as _arc
from ..backends.opengl.nodes.primitives.sector.implementation import reference as _sector
from ..backends.opengl.nodes.primitives.repeat.implementation import reference as _repeat
from ..backends.opengl.nodes.primitives.transform.implementation import reference as _transform
from ..backends.opengl.nodes.primitives.triangle.implementation import reference as _triangle
from ..backends.opengl.nodes.primitives.bezier.implementation import reference as _bezier

__all__ = ["RULES"]
RULES = {
    "abs": _abs,
    "arc": _arc,
    "bezier": _bezier,
    "capsule": _capsule,
    "circle": _circle,
    "ellipse": _ellipse,
    "grid": _grid,
    "half_plane": _half_plane,
    "interpolation": _interpolate,
    "intersection": _intersection,
    "line": _line,
    "masked_union": _masked_union,
    "rect": _rect,
    "repeat": _repeat,
    "ring": _ring,
    "sector": _sector,
    "smooth_min": _smooth_min,
    "subtract": _subtract,
    "transform": _transform,
    "triangle": _triangle,
    "union": _union,
}
