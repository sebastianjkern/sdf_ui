from gpu_sdf.compose import RECT, fragment_shader
from gpu_sdf.shader import *
from gpu_sdf.window import *


class GLSLTypes:
    def __init__(self):
        self.FLOAT = (0.0,)
        self.VEC2 = (0.0, 0.0)
        self.VEC3 = (0.0, 0.0, 0.0)
        self.VEC4 = (0.0, 0.0, 0.0, 0.0)


class Uniforms:
    def __init__(self):
        _t = GLSLTypes()

        self.ANTIALIASING_DISTANCE = ["antialiasing_distance", _t.FLOAT]
        self.ELEVATION = ["elevation", _t.FLOAT]
        self.CENTER = ["center", _t.VEC2]
        self.SIZE = ["size", _t.VEC2]
        self.CORNER_RADIUS = ["corner_radius", _t.VEC4]
        self.RADIUS = ["radius", _t.FLOAT]
        self.VERTICAL_STRETCH = ["vertical_stretch", _t.FLOAT]
        self.OBJ_COL = ["obj_col", _t.VEC4]
        self.SHADOW_COL = ["shadow_col", _t.VEC3]


if __name__ == '__main__':
    window = init_glfw(1280, 720, visible=True)

    program = compile_shader_program(fragment_source=fragment_shader(RECT))

    while not glfw.window_should_close(window):
        render(window, program, save_image=False)

    glDeleteProgram(program)

    terminate_glfw(window)
