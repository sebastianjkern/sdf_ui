"""GPU context management, shader loading, and texture allocation."""

__docformat__ = "google"

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import moderngl as mgl
import numpy as np

from ..log import logger
from .shaders import ShaderLibrary

# Re-export the texture bookkeeping helpers so existing imports keep working.
from .texture_utils import (  # noqa: F401
    decrease_tex_registry,
    get_tex_registry,
    show_texture,
    tex_registry,
)


def _triple(value, name):
    items = tuple(value)
    if len(items) == 2:
        items = (*items, 1)
    if len(items) != 3:
        raise ValueError(f"{name} must have two or three values")
    if any(int(item) <= 0 for item in items):
        raise ValueError(f"{name} values must be greater than 0")
    return tuple(int(item) for item in items)


@dataclass(frozen=True)
class DispatchConfig:
    """Compute dispatch configuration for shader execution.

    ``shader_local_size`` must match the ``layout(local_size_*)`` declared by
    the shader. The built-in shaders currently use 16x16x1.
    """

    shader_local_size: Tuple[int, int, int] = (16, 16, 1)
    group_count: Optional[Tuple[int, int, int]] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "shader_local_size",
            _triple(self.shader_local_size, "shader_local_size"),
        )
        if self.group_count is not None:
            object.__setattr__(
                self, "group_count", _triple(self.group_count, "group_count")
            )

    @classmethod
    def from_value(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("dispatch_config must be a DispatchConfig, dict, or None")

    def groups_for_size(self, size):
        if self.group_count is not None:
            return self.group_count

        width, height = size
        group_width, group_height, _group_depth = self.shader_local_size
        return (
            math.ceil(width / group_width),
            math.ceil(height / group_height),
            1,
        )


class Context:
    """
    Represents a rendering context with functionality for managing shaders and textures.

    Args:
    - size (tuple): A tuple representing the size (width, height) of the rendering context.

    Example:
    >>> context = Context((800, 600))
    """

    def __init__(self, size, dispatch_config=None):
        self.size = size
        self._closed = False

        self._mgl_ctx = mgl.create_standalone_context()
        self._shader_library = ShaderLibrary(self._mgl_ctx)
        self._active_render_stats = None
        self.last_render_stats = None

        self.dispatch_config = DispatchConfig.from_value(dispatch_config)
        self.dispatch_groups = self.dispatch_config.groups_for_size(size)

    @property
    def local_size(self):
        """Compatibility alias for the compute shader dispatch group count."""
        return self.dispatch_groups

    @local_size.setter
    def local_size(self, value):
        self.dispatch_groups = _triple(value, "local_size")

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
        image_path = Path(path)
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Unable to load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.flip(img, 0).copy(order="C")
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
        tex = self._mgl_ctx.texture(self.size, 1, dtype="f4")
        tex.filter = mgl.LINEAR, mgl.LINEAR
        stats = getattr(self, "_active_render_stats", None)
        if stats is not None:
            stats.record_texture_allocation("r32f")

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
        stats = getattr(self, "_active_render_stats", None)
        if stats is not None:
            stats.record_texture_allocation("rgba8")

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
