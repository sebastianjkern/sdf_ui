"""Signed-distance texture helpers and boolean composition aliases."""

__docformat__ = "google"

from ..gpu.nodes.common import build
from .texture import TextureNode


class SDFTexture(TextureNode):
    kind = "sdf"

    def fill(
        self,
        fg_color,
        bg_color=(0.0, 0.0, 0.0, 1.0),
        inflate=None,
        inner=None,
        outer=0.0,
    ):
        """Fill this signed distance field with foreground and background colors."""
        is_text = self.op == "text"
        if inflate is None:
            inflate = 0.2 if is_text else 0.0
        return build(
            "fill",
            self,
            fg_color=fg_color,
            bg_color=bg_color,
            inflate=inflate,
            inner=(-0.5 if is_text else -1.5) if inner is None else inner,
            outer=outer,
        )

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
