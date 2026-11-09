"""Helpers for tracking and visualizing materialized textures."""

__docformat__ = "google"

import weakref

import moderngl as mgl
from PIL import Image, ImageOps

from ..log import logger


class Counter:
    def __init__(self, value=0, name="TEX_REGISTRY") -> None:
        self.value = value
        self.name = name
        self.max = 0

    def __add__(self, other):
        if not isinstance(other, int):
            raise TypeError("Texture counts require an integer")

        self.value += other
        self.max = max(self.max, self.value)
        logger().debug(f"Value of {self.name} is {self.value}")
        return self

    def __sub__(self, other):
        if not isinstance(other, int):
            raise TypeError("Texture counts require an integer")

        self.value -= other
        logger().debug(f"Value of {self.name} is {self.value}")
        return self

    def __int__(self):
        return self.value

    def __str__(self) -> str:
        return str(self.value)


tex_registry = Counter(0)
_tracked_textures: dict[int, weakref.finalize] = {}


def register_texture(texture):
    """Track a Canvas allocation without taking ownership of the texture."""
    global tex_registry
    key = id(texture)
    if key in _tracked_textures:
        raise ValueError("Texture is already registered")
    counter = tex_registry

    def finalize():
        _tracked_textures.pop(key, None)
        counter.__sub__(1)

    finalizer = weakref.finalize(texture, finalize)
    _tracked_textures[key] = finalizer
    tex_registry += 1


def decrease_tex_registry(texture):
    """Release the diagnostic count only for a tracked Canvas allocation."""
    logger().debug("Deleted texture...")
    finalizer = _tracked_textures.get(id(texture))
    if finalizer is not None:
        finalizer()


def get_tex_registry():
    print(tex_registry)


def show_texture(tex: mgl.Texture):
    """Display the content of an OpenGL texture."""

    mode = "F"
    if tex.dtype == "f1":
        if tex.components == 3:
            mode = "RGB"
        elif tex.components == 4:
            mode = "RGBA"
        else:
            raise NotImplementedError(
                "the mode for the show_texture function is not implemented"
            )

    image = Image.frombytes(mode, tex.size, tex.read(), "raw")
    image = ImageOps.flip(image)
    image.show()
