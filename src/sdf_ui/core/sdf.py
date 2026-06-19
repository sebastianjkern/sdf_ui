__docformat__ = "google"

from PIL import Image
from moderngl import Texture

from .color import ColorSpaceMode, ColorTexture
from .context import Context, decrease_tex_registry, show_texture
from .shaders import Shaders
from ..log import logger


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

    def partial_derivative(self) -> ColorTexture:
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

        shader = self.context.get_shader(Shaders.PARTIAL_DERIVATIVE)
        shader['destTex'] = 0
        shader['sdf'] = 1

        tex = self.context.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*self.context.local_size)

        logger().debug(f"Running {Shaders.PARTIAL_DERIVATIVE} shader...")

        return ColorTexture(tex, context=self.context, mode=ColorSpaceMode.RGB)


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


