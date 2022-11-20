from pathlib import Path

import moderngl as mgl
import numpy as np
import ttfquery
from PIL import Image
from ttfquery import describe, glyph, glyphquery

from framework.log import logger
from framework.shader import ShaderFileDescriptor, Shaders


def rgb_col(r: int, g: int, b: int, *a):
    values = np.array([r, g, b, *a], dtype=float)
    values /= 255
    return tuple(values)


def hex_col(string: str):
    g = string.lstrip("#")
    col = tuple(int(g[i:i + 2], 16) for i in (0, 2, 4))
    return rgb_col(*col, 255)


class SDF:
    def __init__(self, initial: callable):
        self.tex = initial()

    def show(self):
        image = Image.frombytes('F', self.tex.size, self.tex.read(), "raw")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.show()


class Layer:
    def __init__(self, initial: callable):
        self.tex = initial()

    def show(self):
        image = Image.frombytes("RGBA", self.tex.size, self.tex.read(), "raw")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.show()

    def save(self, name):
        image = Image.frombytes("RGBA", self.tex.size, self.tex.read(), "raw")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save(name)


class Context:
    def __init__(self, size):
        self._size = size

        self._shader_cache = {}
        self._shader_lut = {}

        self.shader = [
            # SDF
            ShaderFileDescriptor(Shaders.RECT, "shader_files/primitives/sdfs/rect.glsl"),
            ShaderFileDescriptor(Shaders.CIRCLE, "shader_files/primitives/sdfs/circle.glsl"),
            ShaderFileDescriptor(Shaders.BEZIER, "shader_files/primitives/sdfs/bezier.glsl"),
            ShaderFileDescriptor(Shaders.LINE, "shader_files/primitives/sdfs/line.glsl"),
            # Booleans
            ShaderFileDescriptor(Shaders.SMOOTH_MIN, "shader_files/primitives/booleans/smin.glsl"),
            ShaderFileDescriptor(Shaders.UNION, "shader_files/primitives/booleans/union.glsl"),
            ShaderFileDescriptor(Shaders.INTERSECTION, "shader_files/primitives/booleans/intersection.glsl"),
            ShaderFileDescriptor(Shaders.INTERPOLATION, "shader_files/primitives/booleans/interpolate.glsl"),
            ShaderFileDescriptor(Shaders.SUBTRACT, "shader_files/primitives/booleans/subtract.glsl"),
            # SDF Transform
            ShaderFileDescriptor(Shaders.ABS, "shader_files/primitives/abs.glsl"),

            # Postprocessing
            ShaderFileDescriptor(Shaders.BLUR_HOR_9, "shader_files/postprocessing/blur9_hor.glsl"),
            ShaderFileDescriptor(Shaders.BLUR_VER_9, "shader_files/postprocessing/blur9_vert.glsl"),
            ShaderFileDescriptor(Shaders.BLUR_VER_13, "shader_files/postprocessing/blur13_vert.glsl"),
            ShaderFileDescriptor(Shaders.BLUR_HOR_13, "shader_files/postprocessing/blur13_hor.glsl"),
            ShaderFileDescriptor(Shaders.TO_LAB, "shader_files/postprocessing/to_lab.glsl"),
            ShaderFileDescriptor(Shaders.TO_RGB, "shader_files/postprocessing/to_rgb.glsl"),
            # Shading
            ShaderFileDescriptor(Shaders.FILL, "shader_files/shading/fill.glsl"),
            ShaderFileDescriptor(Shaders.OUTLINE, "shader_files/shading/outline.glsl"),
            # Layer
            ShaderFileDescriptor(Shaders.LAYER_MASK, "shader_files/layer/layer_mask.glsl"),
            ShaderFileDescriptor(Shaders.OVERLAY, "shader_files/layer/overlay.glsl"),
        ]

        for s in self.shader:
            self._shader_lut[s.name] = s.path

        self._mgl_ctx = mgl.create_standalone_context()

        w, h = size
        gw, gh = 16, 16
        self.local_size = int(w / gw + 0.5), int(h / gh + 0.5), 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _get_shader(self, shader: str):
        if shader not in self._shader_cache.keys():
            path = Path(__file__).parent / self._shader_lut[shader]
            code = open(path).read()

            shader_program = self._mgl_ctx.compute_shader(code)
            self._shader_cache[shader] = shader_program
            logger().debug(f"Compiled and cached {shader} Shader...")

        return self._shader_cache[shader]

    # Generate textures
    def r32f(self):
        tex = self._mgl_ctx.texture(self._size, 1, dtype='f4')
        tex.filter = mgl.LINEAR, mgl.LINEAR
        return tex

    def rgba8(self):
        tex = self._mgl_ctx.texture(self._size, 4)
        tex.filter = mgl.LINEAR, mgl.LINEAR
        return tex

    def smooth_min(self, obj1: SDF, obj2: SDF, k=0.025):
        def initial():
            shader = self._get_shader(Shaders.SMOOTH_MIN)
            shader['destTex'] = 0
            shader['sdf0'] = 1
            shader['sdf1'] = 2
            shader['smoothness'] = k

            tex = self.r32f()
            tex.bind_to_image(0, read=False, write=True)
            obj1.tex.bind_to_image(1, read=True, write=False)
            obj2.tex.bind_to_image(2, read=True, write=False)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.SMOOTH_MIN} shader...")

            return tex

        return SDF(initial=initial)

    def union(self, obj1: SDF, obj2: SDF):
        def initial():
            shader = self._get_shader(Shaders.UNION)
            shader['destTex'] = 0
            shader['sdf0'] = 1
            shader['sdf1'] = 2

            tex = self.r32f()
            tex.bind_to_image(0, read=False, write=True)
            obj1.tex.bind_to_image(1, read=True, write=False)
            obj2.tex.bind_to_image(2, read=True, write=False)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.UNION} shader...")

            return tex

        return SDF(initial=initial)

    def subtract(self, obj1: SDF, obj2: SDF):
        def initial():
            shader = self._get_shader(Shaders.SUBTRACT)
            shader['destTex'] = 0
            shader['sdf0'] = 1
            shader['sdf1'] = 2

            tex = self.r32f()
            tex.bind_to_image(0, read=False, write=True)
            obj1.tex.bind_to_image(1, read=True, write=False)
            obj2.tex.bind_to_image(2, read=True, write=False)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.SUBTRACT} shader...")

            return tex

        return SDF(initial=initial)

    def intersection(self, obj1: SDF, obj2: SDF):
        def initial():
            shader = self._get_shader(Shaders.INTERSECTION)
            shader['destTex'] = 0
            shader['sdf0'] = 1
            shader['sdf1'] = 2

            tex = self.r32f()
            tex.bind_to_image(0, read=False, write=True)
            obj1.tex.bind_to_image(1, read=True, write=False)
            obj2.tex.bind_to_image(2, read=True, write=False)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.INTERSECTION} shader...")

            return tex

        return SDF(initial=initial)

    # Primitives
    def rounded_rect(self, offset: tuple, size: tuple, corner_radius: tuple):
        def initial():
            shader = self._get_shader(Shaders.RECT)
            shader['destTex'] = 0
            shader['offset'] = offset
            shader['size'] = size
            shader['corner_radius'] = corner_radius

            tex = self.r32f()
            tex.bind_to_image(0, read=False, write=True)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.RECT} shader...")

            return tex

        return SDF(initial=initial)

    def disc(self, offset, radius):
        def initial():
            shader = self._get_shader(Shaders.CIRCLE)
            shader["destTex"] = 0
            shader["offset"] = offset
            shader["radius"] = radius

            tex = self.r32f()
            tex.bind_to_image(0, read=False, write=True)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.CIRCLE} shader...")

            return tex

        return SDF(initial=initial)

    def bezier(self, a, b, c):
        def initial():
            shader = self._get_shader(Shaders.BEZIER)
            shader['destTex'] = 0
            shader['a'] = a
            shader['b'] = b
            shader['c'] = c

            tex = self.r32f()
            tex.bind_to_image(0, read=False, write=True)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.BEZIER} shader...")

            return tex

        return SDF(initial=initial)

    def line(self, a, b):
        def initial():
            shader = self._get_shader(Shaders.LINE)
            shader['destTex'] = 0
            shader['a'] = a
            shader['b'] = b

            tex = self.r32f()
            tex.bind_to_image(0, read=False, write=True)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.LINE} shader...")

            return tex

        return SDF(initial=initial)

    # SDF Transform
    def abs(self, sdf: SDF):
        def initial():
            shader = self._get_shader(Shaders.ABS)
            shader['destTex'] = 0
            shader['sdf0'] = 0

            tex = self.r32f()
            tex.bind_to_image(0, read=False, write=True)
            sdf.tex.bind_to_image(1, read=True, write=False)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.ABS} shader...")

            return tex

        return SDF(initial=initial)

    # Shading
    def fill(self, sdf: SDF, fg_color, bg_color, inflate):
        def initial():
            shader = self._get_shader(Shaders.FILL)
            shader['destTex'] = 0
            shader['sdf'] = 1
            shader['inflate'] = inflate
            shader['background'] = bg_color
            shader['color'] = fg_color

            tex = self.rgba8()
            tex.bind_to_image(0, read=False, write=True)
            sdf.tex.bind_to_image(1, read=True, write=False)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.FILL} shader...")

            return tex

        return Layer(initial=initial)

    def outline(self, sdf: SDF, fg_color, bg_color, inflate=0):
        def initial():
            shader = self._get_shader(Shaders.OUTLINE)
            shader['destTex'] = 0
            shader['sdf'] = 1
            shader['background'] = bg_color
            shader['outline'] = fg_color
            shader['inflate'] = inflate

            tex = self.rgba8()
            tex.bind_to_image(0, read=False, write=True)
            sdf.tex.bind_to_image(1, read=True, write=False)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.OUTLINE} shader...")

            return tex

        return Layer(initial=initial)

    def glyph(self, glyph, scale, ox, oy, path="fonts/SFUIDisplay-Bold.ttf"):
        control_points = self._get_glyph(path, glyph)

        union_sdf = None

        for x, shape in enumerate(control_points):
            for y, stroke in enumerate(shape):
                ax, ay = stroke[0][0] * scale + ox, stroke[0][1] * scale + oy
                bx, by = stroke[1][0] * scale + ox, stroke[1][1] * scale + oy
                cx, cy = stroke[2][0] * scale + ox, stroke[2][1] * scale + oy

                a = (ax, ay)
                b = (bx, by)
                c = (cx, cy)

                if not self._collinear(ax, ay, bx, by, cx, cy):
                    bezier_sdf = self.bezier(a, b, c)
                else:
                    bezier_sdf = self.line(a, c)

                if y == 0 and x == 0:
                    union_sdf = bezier_sdf
                else:
                    union_sdf = self.union(union_sdf, bezier_sdf)

        return union_sdf

    # Postprocessing
    def _blur_13(self, layer: Layer, n=0):
        def initial():
            vert = self._get_shader(Shaders.BLUR_VER_13)
            vert['destTex'] = 0
            vert['origTex'] = 1

            hor = self._get_shader(Shaders.BLUR_HOR_13)
            hor['destTex'] = 0
            hor['origTex'] = 1

            tex0 = self.rgba8()
            tex1 = self.rgba8()

            tex0.bind_to_image(0, read=False, write=True)
            layer.tex.bind_to_image(1, read=True, write=False)
            vert.run(*self.local_size)

            tex1.bind_to_image(0, read=False, write=True)
            tex0.bind_to_image(1, read=True, write=False)
            hor.run(*self.local_size)

            for _ in range(n):
                tex0.bind_to_image(0, read=False, write=True)
                tex1.bind_to_image(1, read=True, write=False)
                vert.run(*self.local_size)

                tex1.bind_to_image(0, read=False, write=True)
                tex0.bind_to_image(1, read=True, write=False)
                hor.run(*self.local_size)

            logger().debug(f"Running {Shaders.BLUR_HOR_13} and {Shaders.BLUR_VER_13} shader {n + 1} times ...")

            return tex1

        return Layer(initial=initial)

    def _blur_9(self, layer: Layer, n=0):
        def initial():
            vert = self._get_shader(Shaders.BLUR_VER_9)
            vert['destTex'] = 0
            vert['origTex'] = 1

            hor = self._get_shader(Shaders.BLUR_HOR_9)
            hor['destTex'] = 0
            hor['origTex'] = 1

            tex0 = self.rgba8()
            tex1 = self.rgba8()

            tex0.bind_to_image(0, read=False, write=True)
            layer.tex.bind_to_image(1, read=True, write=False)
            vert.run(*self.local_size)

            tex1.bind_to_image(0, read=False, write=True)
            tex0.bind_to_image(1, read=True, write=False)
            hor.run(*self.local_size)

            for _ in range(n):
                tex0.bind_to_image(0, read=False, write=True)
                tex1.bind_to_image(1, read=True, write=False)
                vert.run(*self.local_size)

                tex1.bind_to_image(0, read=False, write=True)
                tex0.bind_to_image(1, read=True, write=False)
                hor.run(*self.local_size)

            logger().debug(f"Running {Shaders.BLUR_HOR_9} and {Shaders.BLUR_VER_9} shader {n + 1} times ...")

            return tex1

        return Layer(initial=initial)

    def to_rgb(self, layer: Layer):
        def initial():
            shader = self._get_shader(Shaders.TO_RGB)
            shader['destTex'] = 0
            shader['origTex'] = 1

            tex = self.rgba8()
            tex.bind_to_image(0, read=False, write=True)
            layer.tex.bind_to_image(1, read=True, write=False)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.TO_RGB} shader...")

            return tex

        return Layer(initial=initial)

    def to_lab(self, layer: Layer):
        def initial():
            shader = self._get_shader(Shaders.TO_LAB)
            shader['destTex'] = 0
            shader['origTex'] = 1

            tex = self.rgba8()
            tex.bind_to_image(0, read=False, write=True)
            layer.tex.bind_to_image(1, read=True, write=False)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.TO_LAB} shader...")

            return tex

        return Layer(initial=initial)

    def blur(self, layer: Layer, n=0, base=9):
        temp = self.to_rgb(layer)
        if base == 9:
            temp = self._blur_9(temp, n)
        elif base == 13:
            temp = self._blur_13(temp, n)
        else:
            logger().debug("Defaulting to 9x9 kernel size for blurring")
            temp = self._blur_9(temp, n)
        return self.to_lab(temp)

    # Layer blending
    def mask(self, top: Layer, bottom: Layer, mask: Layer):
        def initial():
            shader = self._get_shader(Shaders.LAYER_MASK)
            shader['destTex'] = 0
            shader['tex0'] = 1
            shader['tex1'] = 2
            shader['mask'] = 3

            tex = self.rgba8()
            tex.bind_to_image(0, read=False, write=True)
            bottom.tex.bind_to_image(1, read=True, write=False)
            top.tex.bind_to_image(2, read=True, write=False)
            mask.tex.bind_to_image(3, read=True, write=False)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.LAYER_MASK} shader...")

            return tex

        return Layer(initial=initial)

    def overlay(self, top: Layer, bottom: Layer):
        def initial():
            shader = self._get_shader(Shaders.OVERLAY)
            shader['destTex'] = 0
            shader['tex0'] = 1
            shader['tex1'] = 2

            tex = self.rgba8()
            tex.bind_to_image(0, read=False, write=True)
            top.tex.bind_to_image(1, read=True, write=False)
            bottom.tex.bind_to_image(2, read=True, write=False)
            shader.run(*self.local_size)

            logger().debug(f"Running {Shaders.OVERLAY} shader...")

            return tex

        return Layer(initial=initial)

    @staticmethod
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

    @staticmethod
    def _collinear(x1, y1, x2, y2, x3, y3):
        area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        return area <= 0.000000001


if __name__ == '__main__':
    size = (1920, 1080)

    with Context(size) as ctx:
        rect = ctx.rounded_rect((450, 450), (200, 200), (10, 50, 150, 150))
        disc = ctx.disc((250, 250), 200)

        bezier = ctx.bezier((900, 800), (700, 400), (350, 350))

        union = ctx.union(rect, disc, k=0.025)
        union = ctx.union(bezier, union, k=0.025)

        layer = ctx.fill(union, hex_col("#e9c46a"), hex_col("#2C2D35"), 0)
        blur = ctx._blur_9(layer, 10)

        mask_sdf = ctx.rounded_rect((int(size[0] / 2), int(size[1] / 2)), (int(size[0] / 5), int(size[1] / 3)),
                                    (150, 150, 150, 150))

        mask_layer = ctx.fill(mask_sdf, (.0, .0, .0, 1.0), (1.0, 1.0, 1.0, 1.0), 0)

        overlay_outline = ctx.outline(mask_sdf, (1.0, 1.0, 1.0, .25), (0.75, 0.75, 0.75, 0.0))

        masked = ctx.mask(blur, layer, mask_layer)
        overlayed = ctx.overlay(overlay_outline, masked)

        overlayed.show()
