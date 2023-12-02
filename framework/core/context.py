from pathlib import Path

import moderngl as mgl

from framework.core.log import logger


class Shaders:
    GRID = "grid"
    RECT = "rect"
    CIRCLE = "circle"
    BEZIER = "bezier"
    LINE = "line"

    # Booleans
    SMOOTH_MIN = "smooth_min"
    UNION = "union"
    INTERSECTION = "intersection"
    INTERPOLATION = "interpolation"
    SUBTRACT = "subtract"

    # SDF Transform
    ABS = "abs"

    # Postprocessing
    BLUR_HOR_9 = "blur_hor_9"
    BLUR_VER_9 = "blur_ver_9"
    BLUR_VER_13 = "blur_hor_13"
    BLUR_HOR_13 = "blur_vert_13"
    TO_LAB = "to_lab"
    TO_RGB = "to_rgb"
    DITHERING = "dithering"
    DITHER_1BIT = "dither_1bit"

    # Shading
    FILL = "fill"
    FILL_FROM_TEXTURE = "fill_from_texture"
    OUTLINE = "outline"
    CLEAR_COLOR = "clear_color"
    PERLIN_NOISE = "perlin_noise"
    FILM_GRAIN = "film_grain"

    # Layer
    LAYER_MASK = "layer_mask"
    OVERLAY = "overlay"
    TRANSPARENCY = "transparency"


class ShaderFileDescriptor:
    def __init__(self, name, path):
        self.name = name
        self.path = path


tex_registry = 0


def decrease_tex_registry():
    global tex_registry
    tex_registry -= 1


def get_tex_registry():
    print(tex_registry)


class Context:
    def __init__(self, size):
        self.size = size

        self._shader_cache = {}
        self._shader_lut = {}

        self.shader = [
            # SDF
            ShaderFileDescriptor(Shaders.RECT, "shader_files/primitives/sdfs/rect.glsl"),
            ShaderFileDescriptor(Shaders.CIRCLE, "shader_files/primitives/sdfs/circle.glsl"),
            ShaderFileDescriptor(Shaders.BEZIER, "shader_files/primitives/sdfs/bezier.glsl"),
            ShaderFileDescriptor(Shaders.LINE, "shader_files/primitives/sdfs/line.glsl"),
            ShaderFileDescriptor(Shaders.GRID, "shader_files/primitives/sdfs/grid.glsl"),

            # Booleans
            ShaderFileDescriptor(Shaders.SMOOTH_MIN, "shader_files/primitives/booleans/smin.glsl"),
            ShaderFileDescriptor(Shaders.UNION, "shader_files/primitives/booleans/union.glsl"),
            ShaderFileDescriptor(Shaders.INTERSECTION, "shader_files/primitives/booleans/intersection.glsl"),
            ShaderFileDescriptor(Shaders.INTERPOLATION, "shader_files/primitives/booleans/interpolate.glsl"),
            ShaderFileDescriptor(Shaders.SUBTRACT, "shader_files/primitives/booleans/subtract.glsl"),

            # SDF Transform
            ShaderFileDescriptor(Shaders.ABS, "shader_files/primitives/transforms/abs.glsl"),

            # Postprocessing
            ShaderFileDescriptor(Shaders.BLUR_HOR_9, "shader_files/postprocessing/blur9_hor.glsl"),
            ShaderFileDescriptor(Shaders.BLUR_VER_9, "shader_files/postprocessing/blur9_vert.glsl"),
            ShaderFileDescriptor(Shaders.BLUR_VER_13, "shader_files/postprocessing/blur13_vert.glsl"),
            ShaderFileDescriptor(Shaders.BLUR_HOR_13, "shader_files/postprocessing/blur13_hor.glsl"),
            ShaderFileDescriptor(Shaders.TO_LAB, "shader_files/postprocessing/to_lab.glsl"),
            ShaderFileDescriptor(Shaders.TO_RGB, "shader_files/postprocessing/to_rgb.glsl"),
            ShaderFileDescriptor(Shaders.DITHERING, "shader_files/postprocessing/dithering.glsl"),
            ShaderFileDescriptor(Shaders.DITHER_1BIT, "shader_files/postprocessing/dither_1bit.glsl"),

            # Shading
            ShaderFileDescriptor(Shaders.FILL, "shader_files/shading/fill.glsl"),
            ShaderFileDescriptor(Shaders.OUTLINE, "shader_files/shading/outline.glsl"),
            ShaderFileDescriptor(Shaders.CLEAR_COLOR, "shader_files/shading/clear_color.glsl"),
            ShaderFileDescriptor(Shaders.PERLIN_NOISE, "shader_files/primitives/sdfs/perlin_noise.glsl"),
            ShaderFileDescriptor(Shaders.FILM_GRAIN, "shader_files/shading/film_grain.glsl"),
            ShaderFileDescriptor(Shaders.FILL_FROM_TEXTURE, "shader_files/shading/fill_from_texture.glsl"),

            # Layer
            ShaderFileDescriptor(Shaders.LAYER_MASK, "shader_files/layer/layer_mask.glsl"),
            ShaderFileDescriptor(Shaders.OVERLAY, "shader_files/layer/overlay.glsl"),
            ShaderFileDescriptor(Shaders.TRANSPARENCY, "shader_files/layer/transparency.glsl")
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

    def get_shader(self, shader: str):
        if shader not in self._shader_cache.keys():
            path = Path(__file__).parent / self._shader_lut[shader]
            code = open(path).read()

            shader_program = self._mgl_ctx.compute_shader(code)
            self._shader_cache[shader] = shader_program
            logger().debug(f"Compiled and cached {shader} Shader...")

        return self._shader_cache[shader]

    def percent(self, alpha):
        return alpha / 100 * self.size[0]

    def percent_of_min(self, alpha):
        return alpha / 100 * min(self.size)

    def percent_x(self, alpha):
        return alpha / 100 * self.size[0]

    def percent_y(self, alpha):
        return alpha / 100 * self.size[1]

    # Generate textures
    def r32f(self):
        tex = self._mgl_ctx.texture(self.size, 1, dtype='f4')
        tex.filter = mgl.LINEAR, mgl.LINEAR

        global tex_registry
        tex_registry += 1

        return tex

    def rgba8(self):
        tex = self._mgl_ctx.texture(self.size, 4)
        tex.filter = mgl.LINEAR, mgl.LINEAR

        global tex_registry
        tex_registry += 1

        return tex


# Context management
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