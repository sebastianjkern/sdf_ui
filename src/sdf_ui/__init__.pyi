from typing import Any

from . import core
from .core import ColorTexture, Context, SDFTexture
from .core.api import ColorNamespace, SDFNamespace
from .core.expressions import Expr, cos, param, percent, percent_of_min, percent_x, percent_y, sin

Canvas = Context
sdf: SDFNamespace
color: ColorNamespace
logger: Any
