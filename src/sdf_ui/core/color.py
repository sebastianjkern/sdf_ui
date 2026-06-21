"""Color texture nodes and color-space conversion helpers."""

__docformat__ = "google"

from .plugins.common import build
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

    def to_lab(self):
        if self.mode == ColorSpaceMode.LAB:
            return self

        return build("to_lab", self)

    def to_rgb(self):
        if self.mode == ColorSpaceMode.RGB:
            return self

        return build("to_rgb", self)
