__docformat__ = "google"

import moderngl as mgl
from PIL import Image

from ..log import logger


class Counter:
    def __init__(self, value=0, name="TEX_REGISTRY") -> None:
        self.value = value
        self.name = name
        self.max = 0

    def __del__(self):
        logger().debug(f"Maximum of {self.name} is {self.max}")

    def __add__(self, other):
        if not isinstance(other, int):
            logger().critical("TypeError")

        self.value += other
        self.max = max(self.max, self.value)
        logger().debug(f"Value of {self.name} is {self.value}")
        return self

    def __sub__(self, other):
        if not isinstance(other, int):
            logger().critical("TypeError")

        self.value -= other
        logger().debug(f"Value of {self.name} is {self.value}")
        return self

    def __int__(self):
        return self.value

    def __str__(self) -> str:
        return str(self.value)


tex_registry = Counter(0)


def decrease_tex_registry():
    logger().debug("Deleted texture...")
    global tex_registry
    tex_registry -= 1


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
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.show()
