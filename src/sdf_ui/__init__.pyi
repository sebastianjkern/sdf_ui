from typing import Any

from . import core as core, learning as learning
from .gpu.context import Context as Context, DispatchConfig as DispatchConfig
from .core import ColorTexture as ColorTexture, SDFTexture as SDFTexture
from .core.api import ColorNamespace, SDFNamespace
from .core.expressions import Expr as Expr, cos as cos, input_value as input_value, param as param, percent as percent, percent_of_min as percent_of_min, percent_x as percent_x, percent_y as percent_y, sin as sin

Canvas = Context
sdf: SDFNamespace
color: ColorNamespace
logger: Any
