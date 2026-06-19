__docformat__ = "google"

from dataclasses import dataclass
from pathlib import Path

from ..log import logger


class Shaders:
    """
    Constants representing shader names used by the rendering pipeline.
    """

    GRID = "grid"
    RECT = "rect"
    CIRCLE = "circle"
    BEZIER = "bezier"
    LINE = "line"
    TRIANGLE = "triangle"

    # Booleans
    SMOOTH_MIN = "smooth_min"
    UNION = "union"
    INTERSECTION = "intersection"
    INTERPOLATION = "interpolation"
    SUBTRACT = "subtract"
    MASKED_UNION = "masked_union"

    # SDF Transform
    ABS = "abs"
    REPEAT = "repeat"

    # Postprocessing
    BLUR_HOR_9 = "blur_hor_9"
    BLUR_VER_9 = "blur_ver_9"
    BLUR_VER_13 = "blur_ver_13"
    BLUR_HOR_13 = "blur_hor_13"
    TO_LAB = "to_lab"
    TO_RGB = "to_rgb"
    DITHERING = "dithering"
    DITHER_1BIT = "dither_1bit"
    INVERT = "invert"

    # Shading
    FILL = "fill"
    FILL_FROM_TEXTURE = "fill_from_texture"
    OUTLINE = "outline"
    CLEAR_COLOR = "clear_color"
    PERLIN_NOISE = "perlin_noise"
    FILM_GRAIN = "film_grain"
    PARTIAL_DERIVATIVE = "partial_derivative"

    # Layer
    LAYER_MASK = "layer_mask"
    OVERLAY = "overlay"
    TRANSPARENCY = "transparency"
    MULTIPLY = "multiply"


@dataclass(frozen=True)
class ShaderFileDescriptor:
    """
    Describes a shader file with a stable public name and package-relative path.
    """

    name: str
    path: str


SHADER_FILES = (
    # SDF
    ShaderFileDescriptor(Shaders.RECT, "shader_files/primitives/sdfs/rect.glsl"),
    ShaderFileDescriptor(Shaders.CIRCLE, "shader_files/primitives/sdfs/circle.glsl"),
    ShaderFileDescriptor(Shaders.BEZIER, "shader_files/primitives/sdfs/bezier.glsl"),
    ShaderFileDescriptor(Shaders.LINE, "shader_files/primitives/sdfs/line.glsl"),
    ShaderFileDescriptor(Shaders.GRID, "shader_files/primitives/sdfs/grid.glsl"),
    ShaderFileDescriptor(Shaders.TRIANGLE, "shader_files/primitives/sdfs/triangle.glsl"),

    # Booleans
    ShaderFileDescriptor(Shaders.SMOOTH_MIN, "shader_files/primitives/booleans/smin.glsl"),
    ShaderFileDescriptor(Shaders.UNION, "shader_files/primitives/booleans/union.glsl"),
    ShaderFileDescriptor(Shaders.INTERSECTION, "shader_files/primitives/booleans/intersection.glsl"),
    ShaderFileDescriptor(Shaders.INTERPOLATION, "shader_files/primitives/booleans/interpolate.glsl"),
    ShaderFileDescriptor(Shaders.SUBTRACT, "shader_files/primitives/booleans/subtract.glsl"),
    ShaderFileDescriptor(Shaders.MASKED_UNION, "shader_files/primitives/booleans/masked_union.glsl"),

    # SDF Transform
    ShaderFileDescriptor(Shaders.ABS, "shader_files/primitives/transforms/abs.glsl"),
    ShaderFileDescriptor(Shaders.REPEAT, "shader_files/primitives/transforms/repeat.glsl"),

    # Postprocessing
    ShaderFileDescriptor(Shaders.BLUR_HOR_9, "shader_files/postprocessing/blur9_hor.glsl"),
    ShaderFileDescriptor(Shaders.BLUR_VER_9, "shader_files/postprocessing/blur9_vert.glsl"),
    ShaderFileDescriptor(Shaders.BLUR_VER_13, "shader_files/postprocessing/blur13_vert.glsl"),
    ShaderFileDescriptor(Shaders.BLUR_HOR_13, "shader_files/postprocessing/blur13_hor.glsl"),
    ShaderFileDescriptor(Shaders.TO_LAB, "shader_files/postprocessing/to_lab.glsl"),
    ShaderFileDescriptor(Shaders.TO_RGB, "shader_files/postprocessing/to_rgb.glsl"),
    ShaderFileDescriptor(Shaders.DITHERING, "shader_files/postprocessing/dithering.glsl"),
    ShaderFileDescriptor(Shaders.DITHER_1BIT, "shader_files/postprocessing/dither_1bit.glsl"),
    ShaderFileDescriptor(Shaders.INVERT, "shader_files/postprocessing/invert.glsl"),

    # Shading
    ShaderFileDescriptor(Shaders.FILL, "shader_files/shading/fill.glsl"),
    ShaderFileDescriptor(Shaders.OUTLINE, "shader_files/shading/outline.glsl"),
    ShaderFileDescriptor(Shaders.CLEAR_COLOR, "shader_files/shading/clear_color.glsl"),
    ShaderFileDescriptor(Shaders.PERLIN_NOISE, "shader_files/primitives/sdfs/perlin_noise.glsl"),
    ShaderFileDescriptor(Shaders.FILM_GRAIN, "shader_files/shading/film_grain.glsl"),
    ShaderFileDescriptor(Shaders.FILL_FROM_TEXTURE, "shader_files/shading/fill_from_texture.glsl"),
    ShaderFileDescriptor(Shaders.PARTIAL_DERIVATIVE, "shader_files/shading/partial_derivative.glsl"),

    # Layer
    ShaderFileDescriptor(Shaders.LAYER_MASK, "shader_files/layer/layer_mask.glsl"),
    ShaderFileDescriptor(Shaders.OVERLAY, "shader_files/layer/overlay.glsl"),
    ShaderFileDescriptor(Shaders.TRANSPARENCY, "shader_files/layer/transparency.glsl"),
    ShaderFileDescriptor(Shaders.MULTIPLY, "shader_files/layer/multiply.glsl"),
)

SHADER_REGISTRY = {descriptor.name: descriptor for descriptor in SHADER_FILES}


class ShaderLibrary:
    """
    Loads, validates, compiles, and caches compute shaders for a ModernGL context.
    """

    def __init__(self, mgl_context, base_path=None):
        self._mgl_ctx = mgl_context
        self._base_path = Path(base_path) if base_path else Path(__file__).parent
        self._cache = {}

    def get(self, shader_name: str):
        if shader_name not in SHADER_REGISTRY:
            known = ", ".join(sorted(SHADER_REGISTRY.keys()))
            raise KeyError(f"Unknown shader '{shader_name}'. Known shaders: {known}")

        if shader_name not in self._cache:
            descriptor = SHADER_REGISTRY[shader_name]
            path = self._base_path / descriptor.path

            if not path.exists():
                raise FileNotFoundError(f"Shader '{shader_name}' is registered at '{path}', but no file exists there.")

            code = path.read_text(encoding="utf-8")
            self._cache[shader_name] = self._mgl_ctx.compute_shader(code)
            logger().debug(f"Compiled and cached {shader_name} Shader...")

        return self._cache[shader_name]
