"""Color texture nodes and color-space conversion helpers."""

__docformat__ = "google"

from numbers import Integral

from ..gpu.nodes.common import build
from ..util import hex_col
from .expressions import Expr
from .texture import PostNamespace, WarpableTextureNode


def _transparent_ink(color):
    if isinstance(color, str):
        color = hex_col(color)
    if isinstance(color, (tuple, list)):
        return (*color[:3], 0.0)
    return (*(Expr("index", (color, index)) for index in range(3)), 0.0)


class ColorSpaceMode:
    LAB = "LAB"
    RGB = "RGB"
    LINEAR = "LINEAR"


class ColorTexture(WarpableTextureNode):
    kind = "color"
    _warp_operation = "warp_color"

    def __init__(self, *args, mode=ColorSpaceMode.LAB, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode

    @property
    def post(self):
        return PostNamespace(self)

    def over(self, *layers, cache: bool = True):
        """Composite color layers in order using straight alpha.

        With no layers, return this color node. With several layers, each later
        layer is composited over the accumulated result. Temporary intermediate
        composites are not cached; set ``cache=False`` to leave the final result
        uncached as well.
        """
        linear = any(layer.mode == ColorSpaceMode.LINEAR for layer in (self, *layers))
        result = self.to_linear() if linear else self
        for index, layer in enumerate(layers):
            result = (
                build("hdr_over", result, layer.to_linear()) if linear
                else build("alpha_overlay", result, layer)
            )
            if index + 1 < len(layers):
                result = result.uncached()
        return result if cache else result.uncached()

    def alpha_overlay(self, other):
        """Place another layer over this layer, using linear light for HDR input."""
        return self.over(other)

    def to_linear(self):
        """Decode display color to floating point linear sRGB."""
        if self.mode == ColorSpaceMode.LINEAR:
            return self
        return build("hdr_decode", self.to_rgb())

    def expose(self, stops=0.0):
        """Multiply linear RGB by two raised to the given exposure stops."""
        return build("hdr_expose", self.to_linear(), stops)

    def film_grain(self, amount=0.075, *, size=1.35):
        """Apply centered luminance grain and return a linear RGBA layer.

        Amount ranges from 0 to 1 and size is the grain spacing in rendered
        pixels (at least 1). Both accept lazy parameters. Grain preserves
        alpha and color ratios, with a gentler effect in bright areas.
        """
        return build("hdr_grain", self.to_linear(), amount, size)

    def bloom(self, *, threshold=1.0, amount=0.35, passes: int = 2):
        """Add blurred light above a linear luminance threshold.

        Threshold and amount accept lazy values. Passes is a fixed positive
        integer controlling the number of separable 13-tap blurs.
        """
        if isinstance(passes, bool) or not isinstance(passes, Integral) or passes < 1:
            raise ValueError("bloom passes must be a positive integer")
        source = self.to_linear()
        light = build("hdr_extract", source, threshold).uncached()
        blurred = build("blur_13", light, int(passes) - 1).uncached()
        return build("hdr_glow", source, blurred, amount)

    def finish(self, *, exposure=0.0, tone=1.0, dither=1.0):
        """Encode linear light as sRGB with exposure, tone mapping, and dithering.

        Tone controls the strength of luminance-based Reinhard compression.
        Zero bypasses compression. Dither is measured in 8-bit code values
        and does not affect alpha. Parameters accept lazy expressions.
        """
        return build("hdr_finish", self.expose(exposure).uncached(), tone, dither)

    def mask(self, top, mask, mode="mix"):
        """Blend this layer and ``top`` through a mask using a named mode.

        ``mode`` may be ``mix``, ``multiply``, ``screen``, ``darken``,
        ``lighten``, or ``over``.
        The mask's red channel selects this layer at 1 and ``top`` at 0.
        """
        return build("mask", self, top, mask, mode=mode)

    def layer_mask(self, top, mask, mode="mix"):
        """Alias for :meth:`mask` with an explicit layer-composition name."""
        return build("layer_mask", self, top, mask, mode=mode)

    def to_lab(self):
        if self.mode == ColorSpaceMode.LINEAR:
            return self.to_rgb().to_lab()
        if self.mode == ColorSpaceMode.LAB:
            return self

        return build("to_lab", self)

    def to_rgb(self):
        if self.mode == ColorSpaceMode.LINEAR:
            return build("hdr_finish", self, 0.0, 0.0)
        if self.mode == ColorSpaceMode.RGB:
            return self

        return build("to_rgb", self)
