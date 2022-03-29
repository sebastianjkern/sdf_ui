from typing import Optional

import glfw
from OpenGL.GL import *
from PIL import ImageOps, Image

from sandbox.app import Uniforms


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


def render(window, program, uniforms: Uniforms = None, save_image: Optional[bool] = False):
    width, height = glfw.get_framebuffer_size(window)
    glViewport(0, 0, width, height)

    glDisable(GL_DEPTH_TEST)

    glEnable(GL_BLEND)
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(1.0, 1.0, 1.0, 1.0)

    us = Uniforms()
    us.ELEVATION[1] = (0.0 / height,)
    us.CENTER[1] = (0.5, 0.5)
    us.ANTIALIASING_DISTANCE[1] = (2.0 / width,)
    us.VERTICAL_STRETCH[1] = (height / width,)
    us.SIZE[1] = (0.1, 0.1)
    us.CORNER_RADIUS[1] = (0.015, 0.015, 0.015, 0.015)
    us.OBJ_COL[1] = (0.1, 0.1, 0.1, 1.0)

    glUseProgram(program)
    for uniform in us.__dict__.values():
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
