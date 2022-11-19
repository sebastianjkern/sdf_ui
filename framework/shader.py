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
    BLUR_HOR = "blur_hor"
    BLUR_VER = "blur_ver"
    TO_RGB = "to_rgb"

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
