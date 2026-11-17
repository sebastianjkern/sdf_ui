"""Color texture nodes and color-space conversion helpers."""

__docformat__ = "google"

from ..gpu.nodes.common import build
from .texture import PostNamespace, TextureNode


class ColorSpaceMode:
    LAB = "LAB"
    RGB = "RGB"


class ColorTexture(TextureNode):
    kind = "color"

    def __init__(self, *args, mode=ColorSpaceMode.LAB, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode

    @property
    def post(self):
        return PostNamespace(self)

    def over(self, other):
        return build("alpha_overlay", self, other)

    def mask(self, top, mask, mode="mix"):
        """Blend this layer and ``top`` through a mask using a named mode.

        ``mode`` may be ``mix``, ``multiply``, ``screen``, or ``over``.
        The mask's red channel selects this layer at 1 and ``top`` at 0.
        """
        return build("mask", self, top, mask, mode=mode)

    def layer_mask(self, top, mask, mode="mix"):
        """Alias for :meth:`mask` with an explicit layer-composition name."""
        return build("layer_mask", self, top, mask, mode=mode)

    def warp(self, field, strength=1.0, steps=1):
        """Resample this color layer along a floating-point vector field."""
        return build("warp_color", self, field, strength=strength, steps=steps)

    def to_lab(self):
        if self.mode == ColorSpaceMode.LAB:
            return self

        return build("to_lab", self)

    def to_rgb(self):
        if self.mode == ColorSpaceMode.RGB:
            return self

        return build("to_rgb", self)
