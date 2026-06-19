__docformat__ = "google"

from PIL import Image
from moderngl import Texture

from .context import Context, decrease_tex_registry, show_texture
from .shaders import Shaders
from ..log import logger


class ColorSpaceMode:
    LAB = "LAB"
    RGB = "RGB"

class ColorTexture:
    """
    Represents a color texture with associated methods for image processing and blending operations.

    Args:
    - tex (Texture): The texture object associated with the color data.

    Raises:
    - ValueError: If the context is not set when creating ColorTextures.

    Example:
    >>> color_texture = ColorTexture(tex)
    """

    def __init__(self, tex, context: Context, mode=ColorSpaceMode.LAB):
        self.tex: Texture = tex
        self.mode = mode
        self.context = context

        if context==None:
            raise ValueError("context can't be None")

        logger().debug(f"ColorTexture with mode {self.mode} is generated")

    def __del__(self):
        """
        Destructor method that releases the associated texture and decreases the texture registry count.

        Example:
        >>> color_texture = ColorTexture(...)
        >>> del color_texture
        """
        self.tex.release()
        decrease_tex_registry()

    def _check_type(self, obj):
        """
        Checks if the type of the given object matches the type of the current instance.

        Args:
        - obj: Object to check against the current instance type.

        Raises:
        - TypeError: If the types do not match.

        Example:
        >>> color_texture = ColorTexture(...)
        >>> other_texture = ColorTexture(...)
        >>> color_texture._check_type(other_texture)
        """
        if not type(self) == type(obj):
            raise TypeError(f"{obj} of type {type(obj)} should be {type(self)}")

    def show(self, conversion=True):
        """
        Displays the ColorTexture using an image viewer.

        Args:
        - conversion (bool): Enable the automatic conversion to RGB Color Space if true

        Example:
        >>> color_texture = ColorTexture(...)
        >>> color_texture.show()
        """

        temp = self.to_rgb() if self.mode == ColorSpaceMode.LAB and conversion else self

        show_texture(temp.tex)

    def save(self, name, conversion=True):
        """
        Saves the ColorTexture to an image file.

        Args:
        - name (str): The file name, including the file extension.
        - conversion (bool): Enable the automatic conversion to RGB Color Space if true

        Example:
        >>> color_texture = ColorTexture(...)
        >>> color_texture.save("output.png")
        """
        temp = self.to_rgb() if self.mode == ColorSpaceMode.LAB and conversion else self

        image = Image.frombytes("RGBA", temp.tex.size, temp.tex.read(), "raw")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save(name)

    def blur_9(self, n: int=0):
        """
        Applies a blur effect to the color texture using a 9x9 kernel.

        Args:
        - n (int): Number of iterations for the blur effect.

        Returns:
        A new ColorTexture with the applied blur effect.

        Example:
        >>> color_texture = ColorTexture(...)
        >>> blurred_texture = color_texture.blur_9(n=3)
        """
        vert = self.context.get_shader(Shaders.BLUR_VER_9)
        vert['destTex'] = 0
        vert['origTex'] = 1

        hor = self.context.get_shader(Shaders.BLUR_HOR_9)
        hor['destTex'] = 0
        hor['origTex'] = 1

        tex0 = self.context.rgba8()
        tex1 = self.context.rgba8()

        tex0.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        vert.run(*self.context.local_size)

        tex1.bind_to_image(0, read=False, write=True)
        tex0.bind_to_image(1, read=True, write=False)
        hor.run(*self.context.local_size)

        for _ in range(n):
            tex0.bind_to_image(0, read=False, write=True)
            tex1.bind_to_image(1, read=True, write=False)
            vert.run(*self.context.local_size)

            tex1.bind_to_image(0, read=False, write=True)
            tex0.bind_to_image(1, read=True, write=False)
            hor.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.BLUR_HOR_9} and {Shaders.BLUR_VER_9} shader {n + 1} times ...")

        decrease_tex_registry()
        tex0.release()

        return ColorTexture(tex1, self.context, mode=self.mode)

    def blur_13(self, n: int=0):
        """
        Applies a blur effect to the color texture using a 13x13 kernel.

        Args:
        - n (int): Number of iterations for the blur effect.

        Returns:
        A new ColorTexture with the applied blur effect.

        Example:
        >>> color_texture = ColorTexture(...)
        >>> blurred_texture = color_texture.blur_13(n=3)
        """
        vert = self.context.get_shader(Shaders.BLUR_VER_13)
        vert['destTex'] = 0
        vert['origTex'] = 1

        hor = self.context.get_shader(Shaders.BLUR_HOR_13)
        hor['destTex'] = 0
        hor['origTex'] = 1

        tex0 = self.context.rgba8()
        tex1 = self.context.rgba8()

        tex0.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        vert.run(*self.context.local_size)

        tex1.bind_to_image(0, read=False, write=True)
        tex0.bind_to_image(1, read=True, write=False)
        hor.run(*self.context.local_size)

        for _ in range(n):
            tex0.bind_to_image(0, read=False, write=True)
            tex1.bind_to_image(1, read=True, write=False)
            vert.run(*self.context.local_size)

            tex1.bind_to_image(0, read=False, write=True)
            tex0.bind_to_image(1, read=True, write=False)
            hor.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.BLUR_HOR_13} and {Shaders.BLUR_VER_13} shader {n + 1} times ...")

        decrease_tex_registry()
        tex0.release()

        return ColorTexture(tex1, context=self.context, mode=self.mode)

    # Needs some special treatment because of the two separate components
    def blur(self, n: int=0, base: int=9):
        """
        Applies a blur effect to the color texture.

        Args:
        - n (int): Number of iterations for the blur effect.
        - base (int): Size of the blur kernel. It can be either 9 or 13.

        Returns:
        A new ColorTexture with the applied blur effect.

        Example:
        >>> color_texture = ColorTexture(...)
        >>> blurred_texture = color_texture.blur(n=3, base=13)
        """
        t = self.to_rgb()

        if base == 9:
            temp = t.blur_9(n)
        elif base == 13:
            temp = t.blur_13(n)
        else:
            logger().debug("Defaulting to 9x9 kernel size for blurring")
            temp = t.blur_9(n)

        return temp.to_lab()

    # Basically the whole color processing stage is performed 
    # in LAB Color Space, because of the superior color
    # interpolation capabilities of LAB in comparison to RGB
    def to_lab(self):
        """
        Converts the color texture to LAB color space.

        Returns:
        A new ColorTexture in LAB color space.

        Example:
        >>> color_texture = ColorTexture(...)
        >>> lab_texture = color_texture.to_lab()
        """
        if self.mode == ColorSpaceMode.LAB:
            return self
        
        shader = self.context.get_shader(Shaders.TO_LAB)
        shader['destTex'] = 0
        shader['origTex'] = 1

        tex = self.context.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.TO_LAB} shader...")

        return ColorTexture(tex, context=self.context, mode=ColorSpaceMode.LAB)

    def to_rgb(self):
        """
        Converts the color texture to RGB format.

        Returns:
        A new ColorTexture in RGB format.

        Example:
        >>> color_texture = ColorTexture(...)
        >>> rgb_texture = color_texture.to_rgb()
        """
        if self.mode == ColorSpaceMode.RGB:
            return self
        
        shader = self.context.get_shader(Shaders.TO_RGB)
        shader['destTex'] = 0
        shader['origTex'] = 1

        tex = self.context.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.TO_RGB} shader...")

        return ColorTexture(tex, context=self.context, mode=ColorSpaceMode.RGB)

    # Some fun with image processing stuff
    def dithering(self):
        """
        Applies dithering to the color texture.

        Returns:
        A new ColorTexture with dithering applied.

        Example:
        >>> color_texture = ColorTexture(...)
        >>> dithered_texture = color_texture.dithering()
        """

        shader = self.context.get_shader(Shaders.DITHERING)
        shader['destTex'] = 0
        shader['origTex'] = 1

        tex = self.context.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.DITHERING} shader...")

        return ColorTexture(tex, context=self.context, mode=self.mode)

    # Black white dithering
    def dither_1bit(self):
        """
        Applies 1-bit dithering to the color texture.

        Returns:
        A new ColorTexture with 1-bit dithering applied.

        Example:
        >>> color_texture = ColorTexture(...)
        >>> dithered_texture = color_texture.dither_1bit()
        """

        shader = self.context.get_shader(Shaders.DITHER_1BIT)
        shader['destTex'] = 0
        shader['origTex'] = 1

        tex = self.context.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.DITHERING} shader...")

        return ColorTexture(tex, context=self.context, mode=self.mode)

    def mask(self, top, mask):
        """
        Applies a mask to blend two color textures.

        Args:
        - top: The ColorTexture to blend on top.
        - mask: The ColorTexture serving as the mask.

        Returns:
        A new ColorTexture resulting from the masked blending operation.

        Example:
        >>> color_texture_base = ColorTexture(...)
        >>> color_texture_top = ColorTexture(...)
        >>> mask_texture = ColorTexture(...)
        >>> masked_texture = color_texture_base.mask(top=color_texture_top, mask=mask_texture)
        """
        self._check_type(top)
        self._check_type(mask)

        shader = self.context.get_shader(Shaders.LAYER_MASK)
        shader['destTex'] = 0
        shader['tex0'] = 1
        shader['tex1'] = 2
        shader['mask'] = 3

        tex = self.context.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        top.tex.bind_to_image(2, read=True, write=False)
        mask.tex.bind_to_image(3, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.LAYER_MASK} shader...")

        return ColorTexture(tex, context=self.context, mode=self.mode)
    
    def multiply(self, other):
        """
        Performs color multiplication with another color texture.

        Args:
        - other: The ColorTexture to multiply.

        Returns:
        A new ColorTexture resulting from the color multiplication operation.

        Example:
        >>> color_texture_1 = ColorTexture(...)
        >>> color_texture_2 = ColorTexture(...)
        >>> multiplied_texture = color_texture_1.multiply(color_texture_2)
        """
        self._check_type(other)

        shader = self.context.get_shader(Shaders.MULTIPLY)
        shader['destTex'] = 0
        shader['mask1'] = 1
        shader['mask2'] = 2

        tex = self.context.rgba8()
        tex.bind_to_image(0, read=False, write=True)

        self.tex.bind_to_image(1, read=True, write=False)
        other.tex.bind_to_image(2, read=True, write=False)

        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.MULTIPLY} shader...")

        return ColorTexture(tex, context=self.context, mode=self.mode)

    def alpha_overlay(self, other):
        """
        Performs alpha overlay blending with another color texture.

        Args:
        - other: The ColorTexture to overlay.

        Returns:
        A new ColorTexture resulting from the alpha overlay operation.

        Example:
        >>> color_texture_1 = ColorTexture(...)
        >>> color_texture_2 = ColorTexture(...)
        >>> overlay_texture = color_texture_1.alpha_overlay(color_texture_2)
        """
        self._check_type(other)

        shader = self.context.get_shader(Shaders.OVERLAY)
        shader['destTex'] = 0
        shader['tex0'] = 1
        shader['tex1'] = 2

        tex = self.context.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        other.tex.bind_to_image(1, read=True, write=False)
        self.tex.bind_to_image(2, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.OVERLAY} shader...")

        return ColorTexture(tex, context=self.context, mode=self.mode)

    def transparency(self, alpha):
        """
        Adjusts the transparency of the color texture.

        Args:
        - alpha (float): The transparency value to apply.

        Returns:
        A new ColorTexture with adjusted transparency.

        Example:
        >>> color_texture = ColorTexture(...)
        >>> transparent_texture = color_texture.transparency(alpha=0.5)
        """

        shader = self.context.get_shader(Shaders.TRANSPARENCY)
        shader['destTex'] = 0
        shader['tex0'] = 1
        shader['alpha'] = alpha

        tex = self.context.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.TRANSPARENCY} shader...")

        return ColorTexture(tex, context=self.context, mode=self.mode)
    
    def invert(self):
        """
        Inverts the colors of the color texture.

        Returns:
        A new ColorTexture with inverted colors.

        Example:
        >>> color_texture = ColorTexture(...)
        >>> inverted_texture = color_texture.invert()
        """    

        shader = self.context.get_shader(Shaders.INVERT)
        shader["destTex"] = 0
        shader["origTex"] = 1

        tex = self.context.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.INVERT} shader...")

        return ColorTexture(tex, context=self.context, mode=self.mode)



