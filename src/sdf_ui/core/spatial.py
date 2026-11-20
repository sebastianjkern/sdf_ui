"""Spatial parameter references shared by render nodes and expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .expressions import Expr

if TYPE_CHECKING:
    from .texture import TextureNode

_CHANNELS = {
    "r": 0,
    "red": 0,
    "x": 0,
    "g": 1,
    "green": 1,
    "y": 1,
    "b": 2,
    "blue": 2,
    "z": 2,
    "a": 3,
    "alpha": 3,
    "w": 3,
}


@dataclass(frozen=True)
class TextureChannel:
    """A numeric view of one channel. Reading it does not allocate a texture."""

    texture: TextureNode
    index: int

    def at_pixel(self) -> Expr:
        """Read the channel at the world coordinate of the output pixel."""
        return Expr("texture_sample", (self.texture, self.index, "pixel"))

    def at_cell(self) -> Expr:
        """Read the channel at the center of the innermost repetition cell."""
        return Expr("texture_sample", (self.texture, self.index, "cell"))


def channel(texture: TextureNode, name: str | int) -> TextureChannel:
    index = _CHANNELS.get(name.lower()) if isinstance(name, str) else name
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 4:
        raise ValueError("A texture channel must be a name or an index from 0 to 3")
    if texture.kind == "sdf" and index != 0:
        raise ValueError("An SDF has one numeric channel")
    return TextureChannel(texture, index)


def samples(value: Any):
    """Yield texture sample expressions in a parameter without rendering them."""
    if isinstance(value, Expr):
        if value.op == "texture_sample":
            yield value
        else:
            for arg in value.args:
                yield from samples(arg)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from samples(item)


def is_spatial(value: Any) -> bool:
    return next(samples(value), None) is not None


def is_spatial_node(node) -> bool:
    op = getattr(node, "op", None)
    return (
        op == "spatial_repeat"
        or (op == "output" and spatial_scene(node))
        or any(is_spatial(value) for _, value in getattr(node, "params", ()))
    )


def spatial_scene(node: TextureNode) -> bool:
    """Return whether the unmaterialized subtree has spatial parameters."""
    seen = set()
    pending = [node]
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen or getattr(item, "tex", None) is not None:
            continue
        seen.add(identity)
        if any(is_spatial(value) for _, value in item.params):
            return True
        if item.op == "spatial_repeat":
            return True
        pending.extend(item.inputs)
    return False
