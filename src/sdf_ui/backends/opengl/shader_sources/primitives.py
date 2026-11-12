"""Shared dual-number programs for all standalone primitive descriptors.

Bodies compile into forward shaders and analytic local backward shaders.
``evaluate`` flattens outputs in declaration order; masked union writes distance,
then RGBA. Integer addresses, angular masks, signs and closest roots are frozen
branch decisions. Sampled inputs still flow through ``ld`` for GPU adjoints.
"""

__all__ = ["SOURCES"]

from ..nodes.primitives.abs.implementation import PROGRAM as _ABS
from ..nodes.primitives.circle.implementation import PROGRAM as _CIRCLE
from ..nodes.primitives.line.implementation import BODY as _LINE
from ..nodes.primitives.capsule.implementation import BODY as _CAPSULE
from ..nodes.primitives.half_plane.implementation import BODY as _HALF_PLANE
from ..nodes.primitives.segment_math import GLSL as _SEGMENT
from ..nodes.primitives.ellipse.implementation import BODY as _ELLIPSE
from ..nodes.primitives.smooth_union.implementation import BODY as _SMOOTH_MIN
from ..nodes.primitives.grid.implementation import BODY as _GRID
from ..nodes.primitives.rect.implementation import BODY as _RECT
from ..nodes.primitives.masked_union.implementation import BODY as _MASKED_UNION
from ..nodes.primitives.arc.implementation import BODY as _ARC
from ..nodes.primitives.sector.implementation import BODY as _SECTOR
from ..nodes.primitives.segment_math import MIN3_GLSL as _MIN3
from ..nodes.primitives.repeat.implementation import BODY as _REPEAT
from ..nodes.primitives.transform.implementation import BODY as _TRANSFORM
from ..nodes.primitives.triangle.implementation import BODY as _TRIANGLE
from ..nodes.primitives.bezier.implementation import BODY as _BEZIER
from ..nodes.primitives.ring.implementation import PROGRAM as _RING
from ..nodes.primitives.union.implementation import PROGRAM as _UNION
from ..nodes.primitives.intersection.implementation import PROGRAM as _INTERSECTION
from ..nodes.primitives.subtract.implementation import PROGRAM as _SUBTRACT
from ..nodes.primitives.interpolate.implementation import PROGRAM as _INTERPOLATION

SOURCES = {
    "abs": _ABS.body,
    "circle": _CIRCLE.body,
    "rect": _RECT,
    "ellipse": _ELLIPSE,
    "line": _LINE,
    "capsule": _CAPSULE,
    "half_plane": _HALF_PLANE,
    "ring": _RING.body,
    "arc": _ARC,
    "sector": _SECTOR,
    "grid": _GRID,
    "triangle": _TRIANGLE,
    "bezier": _BEZIER,
    "union": _UNION.body,
    "intersection": _INTERSECTION.body,
    "subtract": _SUBTRACT.body,
    "interpolation": _INTERPOLATION.body,
    "smooth_min": _SMOOTH_MIN,
    "masked_union": _MASKED_UNION,
    "repeat": _REPEAT,
    "transform": _TRANSFORM,
}
