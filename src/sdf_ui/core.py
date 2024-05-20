__docformat__ = "google"

import math

from PIL import Image
from moderngl import Texture

from .context import Shaders, decrease_tex_registry, Context, show_texture
from .log import logger

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
        
    def _mode_equal(self, obj):
        if self.mode is not obj.mode:
            raise TypeError(f"Modes of {obj} and {self} do not match")

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
        
        t = self.to_rgb() if self.mode is not ColorSpaceMode.RGB else self

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
        self.tex.bind_to_image(1, read=False, write=True)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.INVERT} shader...")

        return ColorTexture(tex, context=self.context, mode=self.mode)


class SDFTexture:
    """
    Base class for everything in the sdf stage of the rendering pipeline.

    Wrapper class around a 32 bit float texture in the dimension of the rendering size.

    Contains the functions that can be executed on one or more SDF textures.
    """

    def __init__(self, tex, context: Context):
        self.tex: Texture = tex
        self.context = context

        if context==None:
            raise ValueError("context can't be None")


    def __del__(self):
        """
        Destructor for cleaning up resources. Releases the associated texture and decreases the texture registry count.
        """
        self.tex.release()
        decrease_tex_registry()

    def __or__(self, other):
        """
        Perform a union operation with another instance.

        Args:
        - other: Another instance of YourClassName

        Returns:
        A new instance representing the union of the current instance and 'other'.
        """
        return self.union(other)

    def __and__(self, other):
        """
        Perform an intersection operation with another instance.

        Args:
        - other: Another instance of YourClassName

        Returns:
        A new instance representing the intersection of the current instance and 'other'.
        """
        return self.intersection(other)

    def __sub__(self, other):
        """
        Perform a set difference operation with another instance.

        Args:
        - other: Another instance of YourClassName

        Returns:
        A new instance representing the set difference (subtraction) of 'other' from the current instance.
        """
        return self.subtract(other)

    def _check_type(self, obj):
        """
        Check if the given object has the same type as the current instance.

        Args:
        - obj: An object to be checked for type compatibility.

        Raises:
        - TypeError: If the type of 'obj' does not match the type of the current instance.

        Example:
        >>> instance = SDFTexture()
        >>> other_instance = SDFTexture()
        >>> instance._check_type(other_instance)  # This will pass without raising an error.
        >>> some_object = SomeOtherClass()
        >>> instance._check_type(some_object)  # This will raise a TypeError.
        """
        if not type(self) == type(obj):
            raise TypeError(f"{obj} of type {type(obj)} should be {type(self)}")

    def show(self):
        """
        Displays the signed distance field (SDF) texture using an image viewer.

        Example:
        >>> sdf_texture = SDFTexture(...)
        >>> sdf_texture.show()
        """
        show_texture(self.tex)

    def save(self, name="./image.png"):
        """
        Saves the signed distance field (SDF) texture to an image file.

        Args:
        - name (str): The name of the image file to save. Default is "./image.png".

        Example:
        >>> sdf_texture = SDFTexture(...)
        >>> sdf_texture.save(name="output_image.png")
        """
        image = Image.frombytes("F", self.tex.size, self.tex.read(), "raw")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image = image.convert("L")
        image.save(name)

    # Boolean operators
    def smooth_union(self, other, k=0.025):
        """
        Performs a smooth union operation with another signed distance field (SDF) texture.

        Args:
        - other: The SDF texture to perform the smooth union with.
        - k (float): The smoothness parameter. Default is 0.025.

        Returns:
        A new SDFTexture resulting from the smooth union operation.

        Example:
        >>> sdf_texture_1 = SDFTexture(...)
        >>> sdf_texture_2 = SDFTexture(...)
        >>> smooth_union_texture = sdf_texture_1.smooth_union(sdf_texture_2, k=0.05)
        """
        self._check_type(other)

        shader = self.context.get_shader(Shaders.SMOOTH_MIN)
        shader['destTex'] = 0
        shader['sdf0'] = 1
        shader['sdf1'] = 2
        shader['smoothness'] = k

        tex = self.context.r32f()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        other.tex.bind_to_image(2, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.SMOOTH_MIN} shader...")

        return SDFTexture(tex, context=self.context)

    def union(self, other):
        """
        Performs a union operation with another signed distance field (SDF) texture.

        Args:
        - other: The SDF texture to union with.

        Returns:
        A new SDFTexture resulting from the union operation.

        Example:
        >>> sdf_texture_1 = SDFTexture(...)
        >>> sdf_texture_2 = SDFTexture(...)
        >>> union_texture = sdf_texture_1.union(sdf_texture_2)
        """
        self._check_type(other)

        shader = self.context.get_shader(Shaders.UNION)
        shader['destTex'] = 0
        shader['sdf0'] = 1
        shader['sdf1'] = 2

        tex = self.context.r32f()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        other.tex.bind_to_image(2, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.UNION} shader...")

        return SDFTexture(tex, context=self.context)
    
    def masked_union(self, other):
        """
        Performs a masked union operation with another signed distance field (SDF) texture.
        Returns both the resulting SDF texture and a color mask.

        Args:
        - other: The SDF texture to perform the masked union with.

        Returns:
        A tuple containing the resulting SDFTexture and ColorTexture representing the masked union.

        Example:
        >>> sdf_texture_1 = SDFTexture(...)
        >>> sdf_texture_2 = SDFTexture(...)
        >>> sdf_result, mask_result = sdf_texture_1.masked_union(sdf_texture_2)
        """
        self._check_type(other)

        shader = self.context.get_shader(Shaders.MASKED_UNION)
        shader['destTex'] = 0
        shader['maskTex'] = 1
        shader['sdf0'] = 2
        shader['sdf1'] = 3

        tex = self.context.r32f()
        tex.bind_to_image(0, read=False, write=True)

        mask = self.context.rgba8()
        mask.bind_to_image(1, read=False, write=True)

        self.tex.bind_to_image(2, read=True, write=False)
        other.tex.bind_to_image(3, read=True, write=False)

        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.MASKED_UNION} shader...")

        return SDFTexture(tex, context=self.context), ColorTexture(mask, context=self.context, mode=ColorSpaceMode.RGB)

    def subtract(self, other):
        """
        Performs a set subtraction operation with another signed distance field (SDF) texture.

        Args:
        - other: The SDF texture to subtract.

        Returns:
        A new SDFTexture resulting from the set subtraction operation.

        Example:
        >>> sdf_texture_1 = SDFTexture(...)
        >>> sdf_texture_2 = SDFTexture(...)
        >>> subtracted_texture = sdf_texture_1.subtract(sdf_texture_2)
        """
        self._check_type(other)

        shader = self.context.get_shader(Shaders.SUBTRACT)
        shader['destTex'] = 0
        shader['sdf0'] = 1
        shader['sdf1'] = 2

        tex = self.context.r32f()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        other.tex.bind_to_image(2, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.SUBTRACT} shader...")

        return SDFTexture(tex, context=self.context)

    def intersection(self, other):
        """
        Performs an intersection operation with another signed distance field (SDF) texture.

        Args:
        - other: The SDF texture to intersect with.

        Returns:
        A new SDFTexture resulting from the intersection operation.

        Example:
        >>> sdf_texture_1 = SDFTexture(...)
        >>> sdf_texture_2 = SDFTexture(...)
        >>> intersection_texture = sdf_texture_1.intersection(sdf_texture_2)
        """
        self._check_type(other)

        shader = self.context.get_shader(Shaders.INTERSECTION)
        shader['destTex'] = 0
        shader['sdf0'] = 1
        shader['sdf1'] = 2

        tex = self.context.r32f()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        other.tex.bind_to_image(2, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.INTERSECTION} shader...")

        return SDFTexture(tex, context=self.context)

    # Postprocessing
    def abs(self):
        """
        Applies the absolute function to the signed distance field (SDF) texture.

        Returns:
        A new SDFTexture resulting from the absolute function applied to the original SDF texture.

        Example:
        >>> sdf_texture = SDFTexture(...)
        >>> abs_texture = sdf_texture.abs()
        """
        shader = self.context.get_shader(Shaders.ABS)
        shader['destTex'] = 0
        shader['sdf0'] = 1

        tex = self.context.r32f()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.ABS} shader...")

        return SDFTexture(tex, context=self.context)

    def repeat(self, s=15.0):
        """
        Repeats the given sdf_texture

        Returns:
        A new texture that is a repetition of 

        Example:
        >>> sdf_texture = SDFTexture(...)
        >>> repeat_texture = sdf_texture.repeat()
        """
        shader = self.context.get_shader(Shaders.REPEAT)
        shader['destTex'] = 0
        shader['sdf0'] = 1
        shader['repeat'] = s

        tex = self.context.r32f()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.REPEAT} shader...")

        return SDFTexture(tex, context=self.context)

    # SDF -> Color Texture
    def fill(self, fg_color, bg_color, inflate=0.0, inner=-1.5, outer=0.0) -> ColorTexture:
        """
        Converts the signed distance field (SDF) texture to a color texture with specified foreground and background colors.
        The inner and outer parameters control color interpolation.

        Args:
        - fg_color: The foreground color of the filled texture.
        - bg_color: The background color of the filled texture.
        - inflate (float): The amount by which to inflate the filled texture. Default is 0.0.
        - inner (float): The color interpolation value for the inner part of the texture. Default is -1.5.
        - outer (float): The color interpolation value for the outer part of the texture. Default is 0.0.

        Returns:
        A ColorTexture representing the filled texture.

        Example:
        >>> sdf_texture = SDFTexture(...)
        >>> foreground_color = (1.0, 0.0, 0.0, 1.0)  # Red
        >>> background_color = (0.0, 0.0, 1.0, 1.0)  # Blue
        >>> filled_texture = sdf_texture.fill(foreground_color, background_color, inflate=0.2, inner=-1.0, outer=0.5)
        """

        shader = self.context.get_shader(Shaders.FILL)
        shader['destTex'] = 0
        shader['sdf'] = 1
        shader['inflate'] = inflate
        shader['background'] = bg_color
        shader['color'] = fg_color
        shader['first'] = inner
        shader['second'] = outer

        tex = self.context.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.FILL} shader...")

        return ColorTexture(tex, context=self.context, mode=ColorSpaceMode.LAB)

    def fill_from_texture(self, layer: ColorTexture, background=(0.0, 0.0, 0.0, 0.0), inflate=0) -> ColorTexture:
        """
        Fills the signed distance field (SDF) texture based on another color texture, with optional background color and inflation.

        Args:
        - layer: The color texture to use for filling.
        - background: The background color. Default is (0.0, 0.0, 0.0, 0.0).
        - inflate (float): The amount by which to inflate the filled texture. Default is 0.

        Returns:
        A ColorTexture representing the filled texture.

        Example:
        >>> sdf_texture = SDFTexture(...)
        >>> color_texture = ColorTexture(...)
        >>> filled_texture = sdf_texture.fill_from_texture(color_texture, background=(1.0, 1.0, 1.0, 1.0), inflate=0.2)
        """
        shader = self.context.get_shader(Shaders.FILL_FROM_TEXTURE)
        shader['destTex'] = 0
        shader['origTex'] = 1
        shader['sdf'] = 2

        shader['background'] = background
        shader['inflate'] = inflate

        tex = self.context.rgba8()
        tex.bind_to_image(0, write=True, read=False)
        layer.tex.bind_to_image(1, write=False, read=True)
        self.tex.bind_to_image(2, write=False, read=True)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.FILL_FROM_TEXTURE} shader...")

        return ColorTexture(tex, context=self.context, mode=layer.mode)

    def outline(self, fg_color, bg_color, inflate=0.0) -> ColorTexture:
        """
        Generates a ColorTexture with the outline of the signed distance field (SDF) with specified colors.

        Args:
        - fg_color: The foreground color of the outline.
        - bg_color: The background color surrounding the outline.
        - inflate (float): The amount by which to inflate the outline. Default is 0.0.

        Returns:
        A ColorTexture representing the outline of the SDF with the specified colors.

        Example:
        >>> sdf_texture = SDFTexture(...)
        >>> foreground_color = (1.0, 0.0, 0.0, 1.0)  # Red
        >>> background_color = (0.0, 0.0, 1.0, 1.0)  # Blue
        >>> inflated_outline_texture = sdf_texture.outline(foreground_color, background_color, inflate=0.1)
        """

        shader = self.context.get_shader(Shaders.OUTLINE)
        shader['destTex'] = 0
        shader['sdf'] = 1
        shader['background'] = bg_color
        shader['outline'] = fg_color
        shader['inflate'] = inflate

        tex = self.context.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.OUTLINE} shader...")

        return ColorTexture(tex, context=self.context, mode=ColorSpaceMode.LAB)

    # Convenience functions (not necessary, build on top of functions above)
    def generate_mask(self, inflate=0.0, color0=(.0, .0, .0, 1.0), color1=(1.0, 1.0, 1.0, 1.0)) -> ColorTexture:
        """
        Generates a mask based on the signed distance field (SDF) texture, allowing customization of colors and inflation.

        Args:
        - inflate (float): The amount by which to inflate the mask. Default is 0.0.
        - color0: The color of the mask where the SDF is negative or zero.
        - color1: The color of the mask where the SDF is positive.

        Returns:
        A ColorTexture representing the generated mask.

        Example:
        >>> sdf_texture = SDFTexture(...)
        >>> mask_texture = sdf_texture.generate_mask(inflate=0.1, color0=(0.0, 0.0, 0.0, 1.0), color1=(1.0, 1.0, 1.0, 1.0))
        """
        return self.fill(color1, color0, inflate).to_rgb()

    def shadow(self, distance=10, inflate=0, transparency=0.75):
        """
        Generates a shadow effect based on the signed distance field (SDF) texture.

        Args:
        - distance (float): The distance of the shadow. Default is 10.
        - inflate (float): The amount by which to inflate the shadow. Default is 0.
        - transparency (float): The transparency of the shadow. Default is 0.75.

        Returns:
        A ColorTexture representing the shadow effect.

        Example:
        >>> sdf_texture = SDFTexture(...)
        >>> shadow_texture = sdf_texture.shadow(distance=15, inflate=0.2, transparency=0.8)
        """
        return self.generate_mask(inflate=inflate, color1=(0.0, 0.0, 0.0, 0.0)) \
            .blur(base=9, n=distance).transparency(transparency)


def interpolate(ctx: Context, tex0: SDFTexture, tex1: SDFTexture, t=0.5) -> SDFTexture:
    """
    Interpolates between two signed distance field (SDF) textures.

    Args:
    - tex0: The first SDFTexture to interpolate from.
    - tex1: The second SDFTexture to interpolate to.
    - t (float): The interpolation factor. It should be in the range [0, 1],
                 where 0 corresponds to tex0, and 1 corresponds to tex1.
                 Default is 0.5, resulting in a mid-point interpolation.

    Returns:
    A new SDFTexture representing the interpolation between tex0 and tex1.

    Example:
    >>> sdf_texture_0 = SDFTexture(...)
    >>> sdf_texture_1 = SDFTexture(...)
    >>> interpolated_texture = interpolate(sdf_texture_0, sdf_texture_1, t=0.3)
    """
    tex = ctx.rgba8()

    shader = ctx.get_shader(Shaders.INTERPOLATION)
    shader['destTex'] = 0
    shader['sdf0'] = 0
    shader['sdf1'] = 0
    shader['t'] = t

    tex.bind_to_image(0, read=False, write=True)
    tex0.tex.bind_to_image(1, read=True, write=False)
    tex1.tex.bind_to_image(1, read=True, write=False)

    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.INTERPOLATION} shader...")

    return SDFTexture(tex)


# Primitives
def rounded_rect(ctx, center, size, corner_radius, angle=0/360*math.pi) -> SDFTexture:
    """
    Creates a signed distance field (SDF) texture representing a rounded rectangle.

    Args:
    - center: The center coordinates of the rectangle.
    - size: The size of the rectangle (width, height).
    - corner_radius: The radius of the rounded corners.
    - angle (float): The rotation angle of the rectangle in radians. Default is 0.

    Returns:
    A new SDFTexture representing a rounded rectangle.

    Example:
    >>> center = (0.0, 0.0)
    >>> size = (2.0, 1.0)
    >>> corner_radius = 0.3
    >>> angle = 45/360*math.pi
    >>> rounded_rect_texture = rounded_rect(center, size, corner_radius, angle)
    """

    shader = ctx.get_shader(Shaders.RECT)
    shader['destTex'] = 0
    shader['offset'] = center
    shader['size'] = size
    shader['corner_radius'] = corner_radius
    shader['angle'] = angle

    tex = ctx.r32f()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.RECT} shader...")

    return SDFTexture(tex, context=ctx)


def disc(ctx, center, radius) -> SDFTexture:
    """
    Creates a signed distance field (SDF) texture representing a filled disc (circle).

    Args:
    - center: The center coordinates of the disc.
    - radius: The radius of the disc.

    Returns:
    A new SDFTexture representing a filled disc.

    Example:
    >>> center = (0.0, 0.0)
    >>> radius = 1.5
    >>> disc_texture = disc(center, radius)
    """

    shader = ctx.get_shader(Shaders.CIRCLE)
    shader["destTex"] = 0
    shader["offset"] = center
    shader["radius"] = radius

    tex = ctx.r32f()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.CIRCLE} shader...")

    return SDFTexture(tex, context=ctx)


def triangle(ctx, point_1, point_2, point_3) -> SDFTexture:
    """

    Renders a triangle using a signed distance field (SDF) approach.

    Args:
        ctx (SDFContext): The context object for the signed distance field rendering.
        point_1 (tuple): Coordinates of the first vertex of the triangle (x, y).
        point_2 (tuple): Coordinates of the second vertex of the triangle (x, y).
        point_3 (tuple): Coordinates of the third vertex of the triangle (x, y).

    Returns:
        SDFTexture: An SDFTexture object representing the rendered triangle.

    Note:
        The triangle is rendered using the specified vertices and the shader associated with
        Shaders.TRIANGLE. The resulting SDFTexture is returned for further use or analysis.

    Example:
        >>> point1 = (0.0, 0.0)
        >>> point2 = (1.0, 0.0)
        >>> point3 = (0.5, 1.0)
        >>> sdf_texture = triangle(ctx, point1, point2, point3)
    """

    shader = ctx.get_shader(Shaders.TRIANGLE)
    shader["destTex"] = 0
    shader["point0"] = point_1
    shader["point1"] = point_2
    shader["point2"] = point_3

    tex = ctx.r32f()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.TRIANGLE} shader...")

    return SDFTexture(tex=tex, context=ctx)

def bezier(ctx, a, b, c) -> SDFTexture:
    """
    Creates a signed distance field (SDF) texture representing a quadratic Bezier curve.

    Args:
    - a: The starting point of the Bezier curve.
    - b: The control point influencing the curvature of the curve.
    - c: The ending point of the Bezier curve.

    Returns:
    A new SDFTexture representing a quadratic Bezier curve.

    Example:
    >>> a = (0.0, 0.0)
    >>> b = (1.0, 2.0)
    >>> c = (2.0, 0.0)
    >>> bezier_texture = bezier(a, b, c)
    """

    shader = ctx.get_shader(Shaders.BEZIER)
    shader['destTex'] = 0
    shader['a'] = a
    shader['b'] = b
    shader['c'] = c

    tex = ctx.r32f()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.BEZIER} shader...")

    return SDFTexture(tex, context=ctx)


def line(ctx, a, b) -> SDFTexture:
    """
    Creates a signed distance field (SDF) texture representing a straight line segment.

    Args:
    - a: The starting point of the line segment.
    - b: The ending point of the line segment.

    Returns:
    A new SDFTexture representing a straight line.

    Example:
    >>> a = (0.0, 0.0)
    >>> b = (2.0, 1.0)
    >>> line_texture = line(a, b)
    """

    shader = ctx.get_shader(Shaders.LINE)
    shader['destTex'] = 0
    shader['a'] = a
    shader['b'] = b

    tex = ctx.r32f()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.LINE} shader...")

    return SDFTexture(tex, context=ctx)


def grid(ctx, offset, size, angle=0/360*math.pi) -> SDFTexture:
    """
    Creates a signed distance field (SDF) texture representing a grid.

    Args:
    - offset: The center coordinates of the grid.
    - size: The size of each cell in the grid (width, height).
    - angle (float): The rotation angle of the grid in radians. Default is 0.

    Returns:
    A new SDFTexture representing a grid.

    Example:
    >>> offset = (0.0, 0.0)
    >>> size = (1.0, 1.0)
    >>> angle = 45/360*math.pi
    >>> grid_texture = grid(offset, size, angle)
    """

    shader = ctx.get_shader(Shaders.GRID)
    shader['destTex'] = 0
    shader['grid_size'] = size
    shader['offset'] = offset
    shader['angle'] = angle

    tex = ctx.r32f()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.GRID} shader...")

    return SDFTexture(tex, context=ctx)


def clear_color(ctx: Context, color) -> ColorTexture:
    """
    Creates a color texture filled with a specified color.

    Args:
    - color: The color to fill the texture with. Should be in RGBA format.

    Returns:
    A new ColorTexture filled with the specified color.

    Example:
    >>> color = (1.0, 0.0, 0.0, 1.0)  # Red color
    >>> clear_color_texture = clear_color(color)
    """
    shader = ctx.get_shader(Shaders.CLEAR_COLOR)
    shader['destTex'] = 0
    shader['color'] = color

    tex = ctx.rgba8()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.CLEAR_COLOR} shader...")

    return ColorTexture(tex, context=ctx, mode=ColorSpaceMode.LAB)


def perlin_noise(ctx: Context) -> ColorTexture:
    """
    Generates Perlin noise and returns it as a color texture.

    Returns:
    A new ColorTexture representing Perlin noise.

    Example:
    >>> perlin_texture = perlin_noise()
    """
    shader = ctx.get_shader(Shaders.PERLIN_NOISE)
    shader['destTex'] = 0

    tex = ctx.rgba8()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.PERLIN_NOISE} shader...")

    return ColorTexture(tex, context=ctx, mode=ColorSpaceMode.RGB)


def film_grain(ctx: Context) -> ColorTexture:
    """
    Generates a film grain effect and returns it as a color texture.

    Returns:
    A new ColorTexture representing a film grain effect.

    Example:
    >>> film_grain_texture = film_grain()
    """
    tex = ctx.rgba8()

    shader = ctx.get_shader(Shaders.FILM_GRAIN)
    shader['destTex'] = 0

    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.FILM_GRAIN} shader...")

    return ColorTexture(tex, context=ctx, mode=ColorSpaceMode.RGB)


def linear_gradient(context: Context, a, b, color1, color2):
    """
    Generates a linear gradient between two points and returns it as a ColorTexture.

    Args:
    - a: The starting point of the gradient.
    - b: The ending point of the gradient.
    - color1: The color at the starting point of the gradient.
    - color2: The color at the ending point of the gradient.

    Returns:
    A ColorTexture representing a linear gradient between color1 and color2.

    Example:
    >>> start_point = (0.0, 0.0)
    >>> end_point = (1.0, 1.0)
    >>> color_at_start = (1.0, 0.0, 0.0, 1.0)  # Red
    >>> color_at_end = (0.0, 0.0, 1.0, 1.0)    # Blue
    >>> gradient_texture = linear_gradient(start_point, end_point, color_at_start, color_at_end)
    """

    ax, ay = a
    bx, by = b

    dx = ax - bx
    dy = ay - by

    cx = ax + dx
    cy = ax - dy

    ex = ax - dx
    ey = ay + dy

    return line(context, (cx, cy), (ex, ey)).abs() \
        .fill(color1, color2, 0, inner=0, outer=math.sqrt(dx * dx + dy * dy))


def radial_gradient(context: Context, a, color1, color2, inner=0, outer=100):
    """
    Generates a radial gradient centered at a point and returns it as a ColorTexture.

    Args:
    - context: The sdf_ui rendering context
    - a: The center point of the radial gradient.
    - color1: The color at the center of the gradient.
    - color2: The color at the outer edge of the gradient.
    - inner (float): The inner radius of the gradient. Default is 0.
    - outer (float): The outer radius of the gradient. Default is 100.

    Returns:
    A ColorTexture representing a radial gradient between color1 and color2.

    Example:
    >>> center_point = (0.0, 0.0)
    >>> color_at_center = (1.0, 0.0, 0.0, 1.0)  # Red
    >>> color_at_outer = (0.0, 0.0, 1.0, 1.0)    # Blue
    >>> gradient_texture = radial_gradient(center_point, color_at_center, color_at_outer)
    """
    return disc(context, a, 0).fill(color1, color2, 0, inner=inner, outer=outer)
