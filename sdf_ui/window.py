from typing import Optional

import glfw
from OpenGL.GL import *
from PIL import ImageOps, Image

from sdf_ui.shader import ShaderProgram, LINE, RECT, CIRCLE


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
        self.COLOR = ["obj_col", _t.VEC4]
        self.SHADOW_COL = ["shadow_col", _t.VEC3]


class ObjectDescriptor:
    def __init__(self, t: str):
        self.uniforms = Uniforms()
        self.shader_name = t


def get_shader(t):
    if t == RECT:
        return ShaderProgram().rect
    if t == LINE:
        return ShaderProgram().line
    if t == CIRCLE:
        return ShaderProgram().circle
    return -1


def init_glfw(width, height, visible: bool = True):
    if not glfw.init():
        raise ValueError("Failed to init glfw")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

    glfw.window_hint(glfw.VISIBLE, visible)
    win = glfw.create_window(width, height, "SDF GPU Renderer", None, None)

    if not win:
        glfw.terminate()
        raise ValueError("Failed to create glfw window")

    glfw.make_context_current(win)
    glfw.swap_interval(1)

    print(glGetString(GL_VERSION).decode("UTF-8"))

    return win


def terminate_glfw(window):
    glfw.destroy_window(window)
    glfw.terminate()


def set_uniform(location, v):
    if isinstance(v, float):
        v = (v,)

    lut = {
        1: glUniform1f,
        2: glUniform2f,
        3: glUniform3f,
        4: glUniform4f,
    }

    lut[len(v)](location, *v)


def render(window, objects: list[ObjectDescriptor], save_image: Optional[bool] = False):
    width, height = glfw.get_framebuffer_size(window)
    glViewport(0, 0, width, height)

    glDisable(GL_DEPTH_TEST)

    glEnable(GL_BLEND)
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(1.0, 1.0, 1.0, 1.0)

    for obj in objects:
        obj.uniforms.VERTICAL_STRETCH[1] = (width / height,)
        obj.uniforms.ANTIALIASING_DISTANCE[1] = (2.0 / height,)

        program = get_shader(obj.shader_name)

        glUseProgram(program)
        for uniform in obj.uniforms.__dict__.values():
            location = glGetUniformLocation(program, uniform[0].encode("UTF-8"))
            set_uniform(location, uniform[1])

        glDrawArrays(GL_POINTS, 0, 1)
        glUseProgram(0)

    if save_image:
        data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (width, height), data)
        image = ImageOps.flip(image)  # in my case image is flipped top-bottom for some reason
        image.save('glutout.png', 'PNG')

    glfw.swap_buffers(window)
    glfw.poll_events()
