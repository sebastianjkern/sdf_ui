import math

from PIL import Image

from framework.core.context import get_context, Shaders, decrease_tex_registry
from framework.core.log import logger


class ColorTexture:
    def __init__(self, tex):
        self.tex = tex

    def __del__(self):
        self.tex.release()
        decrease_tex_registry()

    def _check_type(self, obj):
        if not type(self) == type(obj):
            raise TypeError(f"{obj} of type {type(obj)} should be {type(self)}")

    def show(self):
        image = Image.frombytes("RGBA", self.tex.size, self.tex.read(), "raw")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.show()

    def save(self, name):
        image = Image.frombytes("RGBA", self.tex.size, self.tex.read(), "raw")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save(name)

    def blur_9(self, n=0):
        ctx = get_context()

        vert = ctx.get_shader(Shaders.BLUR_VER_9)
        vert['destTex'] = 0
        vert['origTex'] = 1

        hor = ctx.get_shader(Shaders.BLUR_HOR_9)
        hor['destTex'] = 0
        hor['origTex'] = 1

        tex0 = ctx.rgba8()
        tex1 = ctx.rgba8()

        tex0.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        vert.run(*ctx.local_size)

        tex1.bind_to_image(0, read=False, write=True)
        tex0.bind_to_image(1, read=True, write=False)
        hor.run(*ctx.local_size)

        for _ in range(n):
            tex0.bind_to_image(0, read=False, write=True)
            tex1.bind_to_image(1, read=True, write=False)
            vert.run(*ctx.local_size)

            tex1.bind_to_image(0, read=False, write=True)
            tex0.bind_to_image(1, read=True, write=False)
            hor.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.BLUR_HOR_9} and {Shaders.BLUR_VER_9} shader {n + 1} times ...")

        decrease_tex_registry()
        tex0.release()

        return ColorTexture(tex1)

    def blur_13(self, n=0):
        ctx = get_context()

        vert = ctx.get_shader(Shaders.BLUR_VER_13)
        vert['destTex'] = 0
        vert['origTex'] = 1

        hor = ctx.get_shader(Shaders.BLUR_HOR_13)
        hor['destTex'] = 0
        hor['origTex'] = 1

        tex0 = ctx.rgba8()
        tex1 = ctx.rgba8()

        tex0.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        vert.run(*ctx.local_size)

        tex1.bind_to_image(0, read=False, write=True)
        tex0.bind_to_image(1, read=True, write=False)
        hor.run(*ctx.local_size)

        for _ in range(n):
            tex0.bind_to_image(0, read=False, write=True)
            tex1.bind_to_image(1, read=True, write=False)
            vert.run(*ctx.local_size)

            tex1.bind_to_image(0, read=False, write=True)
            tex0.bind_to_image(1, read=True, write=False)
            hor.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.BLUR_HOR_13} and {Shaders.BLUR_VER_13} shader {n + 1} times ...")

        decrease_tex_registry()
        tex0.release()

        return ColorTexture(tex1)

    # Needs some special treatment because of the two separate components
    def blur(self, n=0, base=9):
        t = self.to_rgb()

        if base == 9:
            temp = t.blur_9(n)
        elif base == 13:
            temp = t.blur_13(n)
        else:
            logger().debug("Defaulting to 9x9 kernel size for blurring")
            temp = t.blur_9(n)

        return temp.to_lab()

    def to_lab(self):
        ctx = get_context()

        shader = ctx.get_shader(Shaders.TO_LAB)
        shader['destTex'] = 0
        shader['origTex'] = 1

        tex = ctx.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.TO_LAB} shader...")

        return ColorTexture(tex)

    def to_rgb(self):
        ctx = get_context()

        shader = ctx.get_shader(Shaders.TO_RGB)
        shader['destTex'] = 0
        shader['origTex'] = 1

        tex = ctx.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.TO_RGB} shader...")

        return ColorTexture(tex)

    def dithering(self):
        ctx = get_context()

        shader = ctx.get_shader(Shaders.DITHERING)
        shader['destTex'] = 0
        shader['origTex'] = 1

        tex = ctx.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.DITHERING} shader...")

        return ColorTexture(tex)

    def dither_1bit(self):
        ctx = get_context()

        shader = ctx.get_shader(Shaders.DITHER_1BIT)
        shader['destTex'] = 0
        shader['origTex'] = 1

        tex = ctx.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.DITHERING} shader...")

        return ColorTexture(tex)

    def mask(self, top, mask):
        self._check_type(top)
        self._check_type(mask)

        ctx = get_context()

        shader = ctx.get_shader(Shaders.LAYER_MASK)
        shader['destTex'] = 0
        shader['tex0'] = 1
        shader['tex1'] = 2
        shader['mask'] = 3

        tex = ctx.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        top.tex.bind_to_image(2, read=True, write=False)
        mask.tex.bind_to_image(3, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.LAYER_MASK} shader...")

        return ColorTexture(tex)

    def alpha_overlay(self, other):
        self._check_type(other)

        ctx = get_context()

        shader = ctx.get_shader(Shaders.OVERLAY)
        shader['destTex'] = 0
        shader['tex0'] = 1
        shader['tex1'] = 2

        tex = ctx.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        other.tex.bind_to_image(1, read=True, write=False)
        self.tex.bind_to_image(2, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.OVERLAY} shader...")

        return ColorTexture(tex)

    def transparency(self, alpha):
        ctx = get_context()

        shader = ctx.get_shader(Shaders.TRANSPARENCY)
        shader['destTex'] = 0
        shader['tex0'] = 1
        shader['alpha'] = alpha

        tex = ctx.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.TRANSPARENCY} shader...")

        return ColorTexture(tex)


class SDFTexture:
    def __init__(self, tex):
        self.tex = tex

    def __del__(self):
        self.tex.release()
        decrease_tex_registry()

    def _check_type(self, obj):
        if not type(self) == type(obj):
            raise TypeError(f"{obj} of type {type(obj)} should be {type(self)}")

    def show(self):
        image = Image.frombytes("RGBA", self.tex.size, self.tex.read(), "raw")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.show()

    # Boolean operators
    def smooth_union(self, other, k=0.025):
        self._check_type(other)
        ctx = get_context()

        shader = ctx.get_shader(Shaders.SMOOTH_MIN)
        shader['destTex'] = 0
        shader['sdf0'] = 1
        shader['sdf1'] = 2
        shader['smoothness'] = k

        tex = ctx.r32f()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        other.tex.bind_to_image(2, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.SMOOTH_MIN} shader...")

        return SDFTexture(tex)

    def union(self, other):
        self._check_type(other)
        ctx = get_context()

        shader = ctx.get_shader(Shaders.UNION)
        shader['destTex'] = 0
        shader['sdf0'] = 1
        shader['sdf1'] = 2

        tex = ctx.r32f()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        other.tex.bind_to_image(2, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.UNION} shader...")

        return SDFTexture(tex)

    def subtract(self, other):
        self._check_type(other)
        ctx = get_context()

        shader = ctx.get_shader(Shaders.SUBTRACT)
        shader['destTex'] = 0
        shader['sdf0'] = 1
        shader['sdf1'] = 2

        tex = ctx.r32f()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        other.tex.bind_to_image(2, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.SUBTRACT} shader...")

        return SDFTexture(tex)

    def intersection(self, other):
        self._check_type(other)
        ctx = get_context()

        shader = ctx.get_shader(Shaders.INTERSECTION)
        shader['destTex'] = 0
        shader['sdf0'] = 1
        shader['sdf1'] = 2

        tex = ctx.r32f()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        other.tex.bind_to_image(2, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.INTERSECTION} shader...")

        return SDFTexture(tex)

    # Postprocessing
    def abs(self):
        ctx = get_context()

        shader = ctx.get_shader(Shaders.ABS)
        shader['destTex'] = 0
        shader['sdf0'] = 0

        tex = ctx.r32f()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.ABS} shader...")

        return SDFTexture(tex)

    # SDF -> Color Texture
    def fill(self, fg_color, bg_color, inflate=0.0, inner=-1.5, outer=0.0) -> ColorTexture:
        ctx = get_context()

        shader = ctx.get_shader(Shaders.FILL)
        shader['destTex'] = 0
        shader['sdf'] = 1
        shader['inflate'] = inflate
        shader['background'] = bg_color
        shader['color'] = fg_color
        shader['first'] = inner
        shader['second'] = outer

        tex = ctx.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.FILL} shader...")

        return ColorTexture(tex)

    def fill_from_texture(self, layer: ColorTexture, background=(0.0, 0.0, 0.0, 0.0), inflate=0) -> ColorTexture:
        ctx = get_context()

        shader = ctx.get_shader(Shaders.FILL_FROM_TEXTURE)
        shader['destTex'] = 0
        shader['origTex'] = 1
        shader['sdf'] = 2

        shader['background'] = background
        shader['inflate'] = inflate

        tex = ctx.rgba8()
        tex.bind_to_image(0, write=True, read=False)
        layer.tex.bind_to_image(1, write=False, read=True)
        self.tex.bind_to_image(2, write=False, read=True)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.FILL_FROM_TEXTURE} shader...")

        return ColorTexture(tex)

    def outline(self, fg_color, bg_color, inflate=0.0) -> ColorTexture:
        ctx = get_context()

        shader = ctx.get_shader(Shaders.OUTLINE)
        shader['destTex'] = 0
        shader['sdf'] = 1
        shader['background'] = bg_color
        shader['outline'] = fg_color
        shader['inflate'] = inflate

        tex = ctx.rgba8()
        tex.bind_to_image(0, read=False, write=True)
        self.tex.bind_to_image(1, read=True, write=False)
        shader.run(*ctx.local_size)

        logger().debug(f"Running {Shaders.OUTLINE} shader...")

        return ColorTexture(tex)

    # Convenience functions (not necessary, build on top of functions above)

    def generate_mask(self, inflate=0.0, color0=(.0, .0, .0, 1.0), color1=(1.0, 1.0, 1.0, 1.0)) -> ColorTexture:
        return self.fill(color0, color1, inflate)

    def shadow(self, distance=10, inflate=0, transparency=0.75):
        return self.generate_mask(inflate=inflate, color1=(0.0, 0.0, 0.0, 0.0)) \
            .blur(base=9, n=distance).transparency(transparency)


def interpolate(tex0: SDFTexture, tex1: SDFTexture, t=0.5) -> SDFTexture:
    ctx = get_context()
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
def rounded_rect(center, size, corner_radius) -> SDFTexture:
    ctx = get_context()

    shader = ctx.get_shader(Shaders.RECT)
    shader['destTex'] = 0
    shader['offset'] = center
    shader['size'] = size
    shader['corner_radius'] = corner_radius

    tex = ctx.r32f()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.RECT} shader...")

    return SDFTexture(tex)


def disc(center, radius) -> SDFTexture:
    ctx = get_context()
    shader = ctx.get_shader(Shaders.CIRCLE)
    shader["destTex"] = 0
    shader["offset"] = center
    shader["radius"] = radius

    tex = ctx.r32f()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.CIRCLE} shader...")

    return SDFTexture(tex)


def bezier(a, b, c) -> SDFTexture:
    ctx = get_context()

    shader = ctx.get_shader(Shaders.BEZIER)
    shader['destTex'] = 0
    shader['a'] = a
    shader['b'] = b
    shader['c'] = c

    tex = ctx.r32f()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.BEZIER} shader...")

    return SDFTexture(tex)


def line(a, b) -> SDFTexture:
    ctx = get_context()

    shader = ctx.get_shader(Shaders.LINE)
    shader['destTex'] = 0
    shader['a'] = a
    shader['b'] = b

    tex = ctx.r32f()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.LINE} shader...")

    return SDFTexture(tex)


def grid(offset, size) -> SDFTexture:
    ctx = get_context()

    shader = ctx.get_shader(Shaders.GRID)
    shader['destTex'] = 0
    shader['grid_size'] = size
    shader['offset'] = offset

    tex = ctx.r32f()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.GRID} shader...")

    return SDFTexture(tex)


def clear_color(color) -> ColorTexture:
    ctx = get_context()

    shader = ctx.get_shader(Shaders.CLEAR_COLOR)
    shader['destTex'] = 0
    shader['color'] = color

    tex = ctx.rgba8()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.CLEAR_COLOR} shader...")

    return ColorTexture(tex)


def perlin_noise() -> ColorTexture:
    ctx = get_context()

    shader = ctx.get_shader(Shaders.PERLIN_NOISE)
    shader['destTex'] = 0

    tex = ctx.rgba8()
    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.PERLIN_NOISE} shader...")

    return ColorTexture(tex)


def film_grain() -> ColorTexture:
    ctx = get_context()
    tex = ctx.rgba8()

    shader = ctx.get_shader(Shaders.FILM_GRAIN)
    shader['destTex'] = 0

    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.FILM_GRAIN} shader...")

    return ColorTexture(tex)


def linear_gradient(a, b, color1, color2):
    ax, ay = a
    bx, by = b

    dx = ax - bx
    dy = ay - by

    cx = ax + dx
    cy = ax - dy

    ex = ax - dx
    ey = ay + dy

    return line((cx, cy), (ex, ey)) \
        .fill(color1, color2, 0, inner=0, outer=math.sqrt(dx * dx + dy * dy))


def radial_gradient(a, color1, color2, inner=0, outer=100):
    return disc(a, 0).fill(color1, color2, 0, inner=inner, outer=outer)
