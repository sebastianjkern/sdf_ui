__docformat__ = "google"

import math

import moderngl as mgl

import cv2

import numpy as np

from .shaders import ShaderLibrary
# Re-export the texture bookkeeping helpers so existing imports keep working.
from .texture_utils import decrease_tex_registry, get_tex_registry, show_texture, tex_registry
from ..log import logger


class Context:
    """
    Represents a rendering context with functionality for managing shaders and textures.

    Args:
    - size (tuple): A tuple representing the size (width, height) of the rendering context.

    Example:
    >>> context = Context((800, 600))
    """
    def __init__(self, size):
        self.size = size
        self._closed = False

        self._mgl_ctx = mgl.create_standalone_context()
        self._shader_library = ShaderLibrary(self._mgl_ctx)

        w, h = size
        gw, gh = 16, 16
        self.local_size = math.ceil(w / gw), math.ceil(h / gh), 1

    def __enter__(self):
        """
        Sets the context as the current context for the duration of a 'with' block.

        Example:
        >>> with Context((800, 600)) as context:
        >>>     # context-related operations
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the 'with' block and performs cleanup if needed.

        Example:
        >>> with Context((800, 600)) as context:
        >>>     # context-related operations
        >>> # Exiting the 'with' block
        """
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        if self._closed:
            return

        self._closed = True
        release = getattr(getattr(self, "_mgl_ctx", None), "release", None)
        if callable(release):
            release()

    def render(self, texture, params=None, cache=None):
        from .texture import Renderer

        return Renderer(self, params=params, cache=cache).render(texture)

    def session(self, params=None, cache=None):
        from .texture import Renderer

        return Renderer(self, params=params, cache=cache)

    def texture_from_image(self, path):
        """
        Loads an image from the specified path and creates a texture.

        Args:
        - path (str): The path to the image file.

        Returns:
        A texture object.

        Example:
        >>> context = Context((800, 600))
        >>> texture = context.texture_from_image("path/to/image.png")
        """
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.flip(img, 0).copy(order='C')
        return self._mgl_ctx.texture(img.shape[1::-1], img.shape[2], img)
    

    def get_shader(self, shader: str):
        """
        Retrieves a shader program based on its name, compiling and caching it if necessary.

        Args:
        - shader (str): The name of the shader.

        Returns:
        A shader program object.

        Example:
        >>> context = Context((800, 600))
        >>> shader_program = context.get_shader("blur_hor_9")
        """
        return self._shader_library.get(shader)

    def percent(self, alpha):
        """
        Converts a percentage value to an absolute value based on the width of the rendering context.

        Args:
        - alpha (float): The percentage value.

        Returns:
        The absolute value.

        Example:
        >>> context = Context((800, 600))
        >>> absolute_value = context.percent(50)
        """
        return alpha / 100 * self.size[0]

    def percent_of_min(self, alpha):
        """
        Converts a percentage value to an absolute value based on the minimum dimension of the rendering context.

        Args:
        - alpha (float): The percentage value.

        Returns:
        The absolute value.

        Example:
        >>> context = Context((800, 600))
        >>> absolute_value = context.percent_of_min(25)
        """
        return alpha / 100 * min(self.size)

    def percent_x(self, alpha):
        """
        Converts a percentage value to an absolute value based on the width of the rendering context.

        Args:
        - alpha (float): The percentage value.

        Returns:
        The absolute value.

        Example:
        >>> context = Context((800, 600))
        >>> absolute_value = context.percent_x(75)
        """
        return alpha / 100 * self.size[0]

    def percent_y(self, alpha):
        """
        Converts a percentage value to an absolute value based on the height of the rendering context.

        Args:
        - alpha (float): The percentage value.

        Returns:
        The absolute value.

        Example:
        >>> context = Context((800, 600))
        >>> absolute_value = context.percent_y(30)
        """
        return alpha / 100 * self.size[1]

    # Generate textures
    def r32f(self):
        """
        Creates a floating-point texture (r32f) with the specified size.

        Returns:
        A floating-point texture.

        Example:
        >>> context = Context((800, 600))
        >>> r32f_texture = context.r32f()
        """
        logger().debug("Created r32f texture...")
        tex = self._mgl_ctx.texture(self.size, 1, dtype='f4')
        tex.filter = mgl.LINEAR, mgl.LINEAR

        global tex_registry
        tex_registry += 1

        return tex

    def rgba8(self):
        """
        Creates an RGBA8 texture with the specified size.

        Returns:
        An RGBA8 texture.

        Example:
        >>> context = Context((800, 600))
        >>> rgba8_texture = context.rgba8()
        """
        logger().debug("Created rgba8 texture...")
        tex = self._mgl_ctx.texture(self.size, 4)
        tex.filter = mgl.LINEAR, mgl.LINEAR

        global tex_registry
        tex_registry += 1

        return tex


def init_sdf_ui(size):
    """
    Initializes the SDF UI with the specified size.

    Args:
    - size (tuple): The size of the context.

    Example:
    >>> init_sdf_ui((800, 600))
    """
    global _context
    _context = Context(size)
