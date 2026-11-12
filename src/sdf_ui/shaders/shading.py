"""Analytic dual GLSL bodies for the nine standalone shading shaders.

``SOURCES[descriptor.name]`` defines ``evaluate(ivec2 pos, out D o[8])``.
The first four outputs are unquantized RGBA; the remaining slots are zero.
The shared emitter supplies dual arithmetic, uniforms, image loads, and color
conversion. Image indices follow the descriptor's input declaration order.
These shared programs generate both the native forward and GPU VJP shaders.
"""

__all__ = ["SOURCES"]

from ..core.plugins.shading.glsl import RGBA as _RGBA
from ..core.plugins.shading.fill.implementation import BODY as _FILL
from ..core.plugins.shading.clear_color.implementation import BODY as _CLEAR_COLOR
from ..core.plugins.shading.outline.implementation import BODY as _OUTLINE
from ..core.plugins.shading.fill_from_texture.implementation import BODY as _FILL_FROM_TEXTURE
from ..core.plugins.shading.partial_derivative.implementation import BODY as _PARTIAL_DERIVATIVE
from ..core.plugins.shading.isolines.implementation import BODY as _ISOLINES
from ..core.plugins.shading.film_grain.implementation import BODY as _FILM_GRAIN
from ..core.plugins.shading.perlin_noise.implementation import BODY as _PERLIN_NOISE
from ..core.plugins.shading.light.implementation import BODY as _LIGHT



SOURCES = {
    name: _RGBA + body
    for name, body in {
        "clear_color": _CLEAR_COLOR,
        "fill": _FILL,
        "fill_from_texture": _FILL_FROM_TEXTURE,
        "film_grain": _FILM_GRAIN,
        "isolines": _ISOLINES,
        "light": _LIGHT,
        "outline": _OUTLINE,
        "partial_derivative": _PARTIAL_DERIVATIVE,
        "perlin_noise": _PERLIN_NOISE,
    }.items()
}
