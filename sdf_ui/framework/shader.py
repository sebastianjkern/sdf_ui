class Shaders:
    RECT = "rect"
    CIRCLE = "circle"
    BEZIER = "bezier"
    LINE = "line"

    # Booleans
    SMOOTH_MIN = "smin"
    UNION = "union"
    INTERSECTION = "intersection"
    INTERPOLATION = "interpolation"
    SUBTRACT = "subtract"

    # Postprocessing
    BLUR_HOR = "blur_hor"
    BLUR_VER = "blur_ver"

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
