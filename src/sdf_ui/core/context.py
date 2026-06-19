__docformat__ = "google"

import moderngl as mgl

import cv2

import numpy as np

from PIL import Image

from .shaders import ShaderLibrary, Shaders
from ..log import logger


class Counter:
    """
    Simple counter class with logging capabilities.
    """
    def __init__(self, value=0, name="TEX_REGISTRY") -> None:
        """
        Initializes the counter.

        Args:
        - value (int): Initial value of the counter.
        - name (str, optional): Name of the counter. Defaults to "TEX_REGISTRY".
        """
        self.value = value
        self.name = name

        self.max = 0

    def __del__(self):
        """
        Destructor method that logs the maximum value of the counter.
        """
        logger().debug(f"Maximum of {self.name} is {self.max}")

    def __add__(self, other):
        """
        Adds a value to the counter.

        Args:
        - other (int): Value to add.

        Returns:
        The updated Counter instance.
        """
        if type(other) != int:
            logger().critical("TypeError")

        self.value += other

        self.max = max(self.max, self.value)

        logger().debug(f"Value of {self.name} is {self.value}")

        return self

    def __sub__(self, other):
        """
        Subtracts a value from the counter.

        Args:
        - other (int): Value to subtract.

        Returns:
        The updated Counter instance.
        """
        if type(other) != int:
            logger().critical("TypeError")

        self.value -= other
        
        logger().debug(f"Value of {self.name} is {self.value}")

        return self

    def __int__(self):
        """
        Converts the Counter to an integer.

        Returns:
        The integer value of the Counter.
        """
        return self.value

    def __str__(self) -> str:
        """
        Converts the Counter to a string.

        Returns:
        The string representation of the Counter.
        """
        return str(self.value)

# The tex registry is not necessary to run the program, 
# but it helps checking for correct texture caching and deletion processes.
tex_registry = Counter(0)


def decrease_tex_registry():
    """
    Decreases the texture registry count and logs a debug message.
    """
    logger().debug("Deleted texture...")
    global tex_registry
    tex_registry -= 1


def get_tex_registry():
    """
    Prints the current texture registry count.
    """
    print(tex_registry)


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

        self._mgl_ctx = mgl.create_standalone_context()
        self._shader_library = ShaderLibrary(self._mgl_ctx)

        w, h = size
        gw, gh = 16, 16
        self.local_size = int(w / gw + 0.5), int(h / gh + 0.5), 1

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


def show_texture(tex: mgl.Texture):
    """Display the content of an OpenGL texture.

    Args:
            tex (mgl.Texture): The OpenGL texture to be displayed.

    Raises:
        ValueError: If the provided texture is not of the expected format or size.

    Note:
        This method uses the Pillow library (PIL) to convert the texture data into an image
        and then displays the image using the default image viewer.

    Example:
        Assuming `my_texture` is an instance of the mgl.Texture class:

        >>> show_texture(my_texture)

    """
    
    mode = "F"
    if tex.dtype == "f1":
        if tex.components == 3:
            mode = "RGB"
        elif tex.components == 4:
            mode = "RGBA"
        else:
            raise NotImplementedError("the mode for the show_texture function is not implemented")

    image = Image.frombytes(mode, tex.size, tex.read(), "raw")
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.show()
