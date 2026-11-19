from typing import Any

from . import core as core, learning as learning
from .gpu.context import Context as Context, DispatchConfig as DispatchConfig
from .core import ColorTexture as ColorTexture, FieldTexture as FieldTexture, SDFTexture as SDFTexture
from .core.api import ColorNamespace, SDFNamespace
from .core.field import FieldNamespace
from .core.palette import Palette as Palette, get_palette as get_palette, palettes as palettes
from .core.expressions import Expr as Expr, cos as cos, ease_in_cubic as ease_in_cubic, ease_in_out_sine as ease_in_out_sine, ease_out_cubic as ease_out_cubic, input_value as input_value, param as param, percent as percent, percent_of_min as percent_of_min, percent_x as percent_x, percent_y as percent_y, sin as sin
from .plot import Plot as Plot

Canvas = Context
sdf: SDFNamespace
color: ColorNamespace
field: FieldNamespace
logger: Any
