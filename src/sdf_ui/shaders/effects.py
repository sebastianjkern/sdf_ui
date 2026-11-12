"""Native analytic dual GLSL for layer and postprocessing descriptors.

``SOURCES[name]`` defines ``evaluate(ivec2 pos, out D o[8])`` for the
``gpu_glsl`` runtime. Inputs follow the differentiation contract's declaration
order; the first four output duals are raw RGBA, before texture quantization.
Only the shared GLSL dual helpers implement arithmetic: no CPU evaluation or
finite differences participate in these derivative kernels.
"""

__all__ = ["SOURCES"]

from ..core.plugins.layer.alpha_overlay.implementation import BODY as _OVERLAY
from ..core.plugins.layer.layer_mask.implementation import BODY as _LAYER_MASK
from ..core.plugins.layer.multiply.implementation import BODY as _MULTIPLY
from ..core.plugins.layer.transparency.implementation import BODY as _TRANSPARENCY
from ..core.plugins.postprocessing.invert.implementation import BODY as _INVERT
from ..core.plugins.postprocessing.to_rgb.implementation import BODY as _TO_RGB
from ..core.plugins.postprocessing.to_lab.implementation import BODY as _TO_LAB
from ..core.plugins.postprocessing.dithering.implementation import BODY as _DITHERING
from ..core.plugins.postprocessing.dither_1bit.implementation import BODY as _DITHER_1BIT
from ..core.plugins.postprocessing.blur.implementation import gpu_body as _blur

_STORE = "o[0]=col.x; o[1]=col.y; o[2]=col.z; o[3]=col.w;"


def _evaluate(body):
    return "void evaluate(ivec2 pos, out D o[8]) {\n" + body + "\n" + _STORE + "\n}\n"


SOURCES = {
    "overlay": _OVERLAY,
    "layer_mask": _LAYER_MASK,
    "multiply": _MULTIPLY,
    "transparency": _TRANSPARENCY,
    "invert": _INVERT,
    "to_rgb": _TO_RGB,
    "to_lab": _TO_LAB,
    "dithering": _DITHERING,
    "dither_1bit": _DITHER_1BIT,
}


SOURCES.update(
    {
        "blur_{}_{}".format(direction, taps): _blur(component, count)
        for direction, component in (("hor", "x"), ("ver", "y"))
        for taps, count in ((9, 5), (13, 8))
    }
)
