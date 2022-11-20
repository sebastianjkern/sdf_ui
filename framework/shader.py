class Shaders:
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

    # Shading
    FILL = "fill"
    OUTLINE = "outline"

    # Layer
    LAYER_MASK = "layer_mask"
    OVERLAY = "overlay"


class ShaderFileDescriptor:
    def __init__(self, name, path):
        self.name = name
        self.path = path
