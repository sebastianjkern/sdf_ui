import logging
import math
import random
import time

import ttfquery
from PIL import Image
from ttfquery import describe, glyphquery, glyph

from framework.ascii import convert_image_to_ascii_colored, convert_image_to_ascii
from framework.context import Context, decrease_tex_registry
from framework.log import logger
from framework.shader import Shaders
from framework.util import hex_col

_context: Context


def init_sdf_ui(size):
    global _context
    _context = Context(size)


def set_context(context: Context):
    global _context
    _context = context


def get_context():
    global _context
    return _context


class ColorTexture:
    def __init__(self, tex):
        self.tex = tex

    def __del__(self):
        self.tex.release()
        decrease_tex_registry()

    def _check_type(self, obj):
        if not type(self) == type(obj):
            raise TypeError(f"{obj} of type {type(obj)} should be {type(self)}")

    def print(self, colored=True):
        image = Image.frombytes("RGBA", self.tex.size, self.tex.read(), "raw")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if not colored:
            convert_image_to_ascii(image, 150, 0.43, True, True, True)
        else:
            convert_image_to_ascii_colored(image, 150, 0.43, more_levels=False, invert=False, enhance=True,
                                           as_background=True)

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
    def fill(self, fg_color, bg_color, inflate=0, inner=-1.5, outer=0.0) -> ColorTexture:
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


# helper stuff
def percent(alpha):
    ctx = get_context()

    return alpha / 100 * ctx.size[0]


def percent_of_min(alpha):
    ctx = get_context()

    return alpha / 100 * min(ctx.size)


def _get_glyph(font_file_path, char):
    font = describe.openFont(font_file_path)
    g = glyph.Glyph(ttfquery.glyphquery.glyphName(font, char))
    ttf_contours = g.calculateContours(font)

    def middle(x1, y1, x2, y2):
        return x1 + 0.5 * (x2 - x1), y1 + 0.5 * (y2 - y1)

    interpolated_contours = []

    for contour in ttf_contours:
        interpolated_points = [list(contour[0][0])]

        for i in range(1, len(contour)):
            if contour[i][1] == contour[i - 1][1]:
                interpolated_points.append(
                    [*middle(
                        contour[i - 1][0][0],
                        contour[i - 1][0][1],
                        contour[i][0][0],
                        contour[i][0][1])])
            interpolated_points.append(list(contour[i][0]))

        interpolated_contours.append(interpolated_points)

    bezier_control_points = []

    for contour in interpolated_contours:
        bezier_curves_control_points = []

        for i in range(0, len(contour) - 1, 2):
            bezier_curves_control_points.append(contour[0 + i:3 + i])

        bezier_control_points.append(bezier_curves_control_points)

    return bezier_control_points


def _collinear(x1, y1, x2, y2, x3, y3):
    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    return area <= 0.000000001


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


def glyph_sdf(glyph, scale, ox, oy, path="fonts/SFUIDisplay-Bold.ttf"):
    control_points = _get_glyph(path, glyph)

    union_sdf = None

    for x, shape in enumerate(control_points):
        for y, stroke in enumerate(shape):
            ax, ay = stroke[0][0] * scale + ox, stroke[0][1] * scale + oy
            bx, by = stroke[1][0] * scale + ox, stroke[1][1] * scale + oy
            cx, cy = stroke[2][0] * scale + ox, stroke[2][1] * scale + oy

            a = (ax, ay)
            b = (bx, by)
            c = (cx, cy)

            if not _collinear(ax, ay, bx, by, cx, cy):
                bezier_sdf = bezier(a, b, c)
            else:
                bezier_sdf = line(a, c)

            if y == 0 and x == 0:
                union_sdf = bezier_sdf
            else:
                union_sdf = union_sdf.union(bezier_sdf)

    return union_sdf


# textures
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


def film_grain() -> ColorTexture:
    ctx = get_context()
    tex = ctx.rgba8()

    shader = ctx.get_shader(Shaders.FILM_GRAIN)
    shader['destTex'] = 0

    tex.bind_to_image(0, read=False, write=True)
    shader.run(*ctx.local_size)

    logger().debug(f"Running {Shaders.FILM_GRAIN} shader...")

    return ColorTexture(tex)


if __name__ == '__main__':
    size = (1920, 1080)

    logger().setLevel(logging.INFO)


    def rand_color():
        red = random.randint(0, 255) / 255
        green = random.randint(0, 255) / 255
        blue = random.randint(0, 255) / 255
        return red, green, blue


    def rand_point():
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])

        return x, y


    COLORS = [
        "#62bb47",
        "#fcb827",
        "#f6821f",
        "#e03a3c",
        "#963d97",
        "#009ddc"
    ]

    context = Context(size)
    set_context(context)

    # Example 1
    col0 = random.choice(COLORS)
    col1 = random.choice(COLORS)
    col2 = random.choice(COLORS)
    col3 = random.choice(COLORS)

    image = clear_color(hex_col(random.choice(COLORS))) \
        .alpha_overlay(
        radial_gradient((100, 100), hex_col(col0, alpha=150), hex_col(col0, alpha=0.0), inner=50, outer=750)) \
        .alpha_overlay(
        radial_gradient((750, 500), hex_col(col1, alpha=255), hex_col(col1, alpha=0.0), inner=50, outer=750)) \
        .alpha_overlay(
        radial_gradient((100, 750), hex_col(col2, alpha=180), hex_col(col2, alpha=0.0), inner=50, outer=750)) \
        .alpha_overlay(
        bezier(rand_point(), rand_point(), rand_point()).fill(hex_col(col3, alpha=150), hex_col(col3, alpha=0),
                                                              inflate=0, inner=0, outer=250)) \
        .alpha_overlay(film_grain().transparency(10 / 255))

    image.to_rgb().show()

    # Example 2

    for _ in range(1):
        start = time.time_ns()

        backdrop = radial_gradient((size[0] / 2, size[1] / 2), hex_col("#004C81", alpha=255),
                                   hex_col("#062A4A", alpha=255), inner=0,
                                   outer=percent_of_min(75)).alpha_overlay(
            grid((10, 10), (50, 50)).fill(hex_col("#5EC6E2", 255), hex_col("#5EC6E2", 0), inflate=.5)) \
            .alpha_overlay(grid((10, 10), (150, 150)).fill(hex_col("#5EC6E2", 255), hex_col("#5EC6E2", 0), inflate=1.5))

        for _ in range(5):
            x, y = random.randint(50, size[0] - 50), random.randint(50, size[1] - 50)
            r = random.randint(10, 100)

            col0 = hex_col(random.choice(COLORS))[:3]
            col1 = (*col0, 1.0)
            col2 = (*col0, 0.0)

            backdrop = backdrop.alpha_overlay(disc((x, y), r).fill(col1, col2, 0))

        gradient = linear_gradient((100, 100), (900, 100), (*rand_color(), 1.0), (*rand_color(), 1.0)).transparency(
            0.45)
        blur = backdrop.blur(n=15, base=13)

        mask_sdf = rounded_rect((int(size[0] / 2), int(size[1] / 2)),
                                (int(size[1] / 3), int(size[1] / 3)),
                                (size[0] / 10, size[0] / 10, size[0] / 10, size[0] / 10))

        mask = mask_sdf.generate_mask()
        outline = mask_sdf.outline((1.0, 1.0, 1.0, .25), (1.0, 1.0, 1.0, 0.0), inflate=-1.5)

        glass_col = (0.75, 0.75, 0.75, 0.75)
        glass = mask_sdf.fill(glass_col, (0.0, 0.0, 0.0, 0.0), 0)

        mask_sdf.fill_from_texture(gradient)

        image = backdrop.mask(blur, mask).alpha_overlay(glass).alpha_overlay(outline).to_rgb().alpha_overlay(
            film_grain().transparency(10 / 255))

        image.show()

        print((time.time_ns() - start) / 1e6)

    # Example 3

    scale = 0.65
    offset_x = 50
    offset_y = 50

    sdf = glyph_sdf("M", scale, offset_x, offset_y, path="../fonts/SFUIDisplay-Bold.ttf")

    bg = clear_color(hex_col("#2C2D35"))
    shadow = sdf.generate_mask(inflate=7.5, color1=(0.0, 0.0, 0.0, 0.0)).blur(n=10, base=13)
    colored_glyph = sdf.fill(hex_col("#e9c46a"), (0.0, 0.0, 0.0, 0.0), 7.5)

    bg.alpha_overlay(shadow).alpha_overlay(colored_glyph).to_rgb().show()
