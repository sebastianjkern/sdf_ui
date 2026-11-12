"""Executable float reference math for standalone shading shader descriptors.

``RULES[name](context, inputs, params)`` consumes GLSL image/uniform names.
The context supplies Tensor pixel grids ``x``, ``y`` and ``size=(width, height)``.
Results are unquantized (height, width, 4) Tensors, with row zero at the bottom.
RGB/LAB conversion follows each shader, rather than imposing a common space.
Continuous expressions retain their autodiff graphs; branch decisions and
modulo cell indices use NumPy booleans/integers and piecewise derivatives.
Noise uses fixed float32 pixel coordinates and has no trainable uniforms.
"""

from ..core.plugins.shading.fill.implementation import reference as _fill
from ..core.plugins.shading.clear_color.implementation import reference as _clear_color
from ..core.plugins.shading.outline.implementation import reference as _outline
from ..core.plugins.shading.fill_from_texture.implementation import reference as _fill_from_texture
from ..core.plugins.shading.partial_derivative.implementation import reference as _partial_derivative
from ..core.plugins.shading.isolines.implementation import reference as _isolines
from ..core.plugins.shading.film_grain.implementation import reference as _film_grain
from ..core.plugins.shading.perlin_noise.implementation import reference as _perlin_noise
from ..core.plugins.shading.light.reference import reference as _light

__all__ = ["RULES"]


RULES = {
    "clear_color": _clear_color,
    "fill": _fill,
    "outline": _outline,
    "fill_from_texture": _fill_from_texture,
    "film_grain": _film_grain,
    "isolines": _isolines,
    "light": _light,
    "partial_derivative": _partial_derivative,
    "perlin_noise": _perlin_noise,
}
