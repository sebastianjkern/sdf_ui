"""Shared dual-number programs for all standalone primitive descriptors.

Bodies compile into forward shaders and analytic local backward shaders.
``evaluate`` flattens outputs in declaration order; masked union writes distance,
then RGBA. Integer addresses, angular masks, signs and closest roots are frozen
branch decisions. Sampled inputs still flow through ``ld`` for GPU adjoints.
"""

__all__ = ["SOURCES"]

from ..core.plugins.primitives.abs.implementation import PROGRAM as _ABS
from ..core.plugins.primitives.circle.implementation import PROGRAM as _CIRCLE
from ..core.plugins.primitives.line.implementation import BODY as _LINE
from ..core.plugins.primitives.capsule.implementation import BODY as _CAPSULE
from ..core.plugins.primitives.half_plane.implementation import BODY as _HALF_PLANE
from ..core.plugins.primitives.segment_math import GLSL as _SEGMENT
from ..core.plugins.primitives.ellipse.implementation import BODY as _ELLIPSE
from ..core.plugins.primitives.smooth_union.implementation import BODY as _SMOOTH_MIN
from ..core.plugins.primitives.grid.implementation import BODY as _GRID
from ..core.plugins.primitives.rect.implementation import BODY as _RECT
from ..core.plugins.primitives.masked_union.implementation import BODY as _MASKED_UNION
from ..core.plugins.primitives.arc.implementation import BODY as _ARC
from ..core.plugins.primitives.sector.implementation import BODY as _SECTOR
from ..core.plugins.primitives.segment_math import MIN3_GLSL as _MIN3
from ..core.plugins.primitives.repeat.implementation import BODY as _REPEAT
from ..core.plugins.primitives.transform.implementation import BODY as _TRANSFORM
from ..core.plugins.primitives.triangle.implementation import BODY as _TRIANGLE
from ..core.plugins.primitives.bezier.implementation import BODY as _BEZIER
from ..core.plugins.primitives.ring.implementation import PROGRAM as _RING
from ..core.plugins.primitives.union.implementation import PROGRAM as _UNION
from ..core.plugins.primitives.intersection.implementation import PROGRAM as _INTERSECTION
from ..core.plugins.primitives.subtract.implementation import PROGRAM as _SUBTRACT
from ..core.plugins.primitives.interpolate.implementation import PROGRAM as _INTERPOLATION

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
