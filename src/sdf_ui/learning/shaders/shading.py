"""Executable float reference math for standalone shading shader descriptors.

``RULES[name](context, inputs, params)`` consumes GLSL image/uniform names.
The context supplies Tensor pixel grids ``x``, ``y`` and ``size=(width, height)``.
Results are unquantized (height, width, 4) Tensors, with row zero at the bottom.
RGB/LAB conversion follows each shader, rather than imposing a common space.
Continuous expressions retain their autodiff graphs; branch decisions and
modulo cell indices use NumPy booleans/integers and piecewise derivatives.
Noise uses fixed float32 pixel coordinates and has no trainable uniforms.
"""

from ...backends.opengl.nodes.shading.clear_color.implementation import (
    reference as _clear_color,
)
from ...backends.opengl.nodes.shading.fill.implementation import reference as _fill
from ...backends.opengl.nodes.shading.fill_from_texture.implementation import (
    reference as _fill_from_texture,
)
from ...backends.opengl.nodes.shading.film_grain.implementation import (
    reference as _film_grain,
)
from ...backends.opengl.nodes.shading.isolines.implementation import (
    reference as _isolines,
)
from ...backends.opengl.nodes.shading.light.reference import reference as _light
from ...backends.opengl.nodes.shading.outline.implementation import (
    reference as _outline,
)
from ...backends.opengl.nodes.shading.partial_derivative.implementation import (
    reference as _partial_derivative,
)
from ...backends.opengl.nodes.shading.perlin_noise.implementation import (
    reference as _perlin_noise,
)

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
