"""Signed-distance texture helpers and boolean composition aliases."""

__docformat__ = "google"

from .plugins.common import build
from .texture import TextureNode


class SDFTexture(TextureNode):
    kind = "sdf"

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __sub__(self, other):
        return self.subtract(other)

    def intersect(self, other):
        return self.intersection(other)

    def mask(
        self, inflate=0.0, color0=(0.0, 0.0, 0.0, 1.0), color1=(1.0, 1.0, 1.0, 1.0)
    ):
        return build(
            "generate_mask", self, inflate=inflate, color0=color0, color1=color1
        )
