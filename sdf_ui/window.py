import sys
import traceback
from typing import Optional

import glfw
from OpenGL.GL import *
from PIL import ImageOps, Image
from namedlist import namedlist

from sdf_ui.shader import get_shader, RECT, CIRCLE, ShaderProgram, LINE
from sdf_ui.util import Singleton
from sdf_ui.util import logger


class GLSLTypeDefaults(Singleton):
    def __init__(self):
        self.FLOAT = (0.0,)
        self.VEC2 = (0.0, 0.0)
        self.VEC3 = (0.0, 0.0, 0.0)
        self.VEC4 = (0.0, 0.0, 0.0, 0.0)


_t = GLSLTypeDefaults()
_Uniforms = namedlist('_Uniforms', [
    ("antialiasing_distance", _t.FLOAT),
    ("elevation", _t.FLOAT),
    ("center", _t.VEC2),
    ("size", _t.VEC2),
    ("corner_radius", _t.VEC4),
    ("radius", _t.FLOAT),
    ("vertical_stretch", _t.FLOAT),
    ("obj_col", _t.VEC4),
    ("shadow_col", _t.VEC3),
    ("a", _t.VEC2),
    ("b", _t.VEC2)])


class RenderObject:
    def __init__(self, t: str):
        self._uniforms = _Uniforms()
        self._shader_name = t

    def elevation(self, value):
        self._uniforms.elevation = value

    def color(self, value):
        self._uniforms.obj_col = value

    def shadow(self, value):
        self._uniforms.shadow_col = value

    def vertical_stretch(self, value):
        self._uniforms.vertical_stretch = value

    def antialiasing_distance(self, value):
        self._uniforms.antialiasing_distance = value

    def get_uniforms(self):
        return self._uniforms

    def get_shader_name(self):
        return self._shader_name


class Rect(RenderObject):
    def __init__(self):
        super(Rect, self).__init__(RECT)

    def center(self, value):
        self._uniforms.center = value

    def size(self, value):
        self._uniforms.size = value

    def corner_radius(self, value):
        self._uniforms.corner_radius = value


class Circle(RenderObject):
    def __init__(self):
        super(Circle, self).__init__(CIRCLE)

    def center(self, value):
        self._uniforms.center = value

    def radius(self, value):
        self._uniforms.radius = value


class Line(RenderObject):
    def __init__(self):
        super(Line, self).__init__(LINE)

    def a(self, value):
        self._uniforms.a = value

    def b(self, value):
        self._uniforms.b = value

    def radius(self, value):
        self._uniforms.radius = value


class SdfUiContext:
    def __init__(self, width=500, height=500, resizable=True, visible=True):
        self.width = width
        self.height = height
        self.resizable = resizable
        self.visible = visible
        self.window = None

    def __enter__(self):
        if not glfw.init():
            msg = ValueError("Failed to init glfw").with_traceback(traceback.print_exc(5))
            logger().error(msg)
            sys.exit(-1)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, False)

        glfw.window_hint(glfw.RESIZABLE, self.resizable)

        glfw.window_hint(glfw.VISIBLE, self.visible)
        self.window = glfw.create_window(self.width, self.height, "SDF GPU Renderer", None, None)

        if not self.window:
            msg = ValueError("Failed to create glfw window").with_traceback(traceback.print_stack())
            logger().error(msg)
            glfw.terminate()
            sys.exit(-1)

        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        logger().debug(glGetString(GL_VERSION).decode("UTF-8"))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        ShaderProgram().cleanup()

        glfw.destroy_window(self.window)
        glfw.terminate()

    def render(self, objects: list[RenderObject], save_image: Optional[bool] = False,
               return_img: Optional[bool] = False, name="glutout.png"):
        glfw.poll_events()

        width, height = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, width, height)

        glDisable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(1.0, 1.0, 1.0, 1.0)

        for obj in objects:
            if height != 0:
                obj.vertical_stretch((width / height,))
                obj.antialiasing_distance((2.0 / height,))

            program = get_shader(obj.get_shader_name())

            glUseProgram(program)
            for uniform_identifier in obj.get_uniforms().__slots__:
                location = glGetUniformLocation(program, uniform_identifier.encode("UTF-8"))
                self.set_uniform(location, getattr(obj.get_uniforms(), uniform_identifier))

            glDrawArrays(GL_POINTS, 0, 1)
            glUseProgram(0)

        data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (width, height), data)
        image = ImageOps.flip(image)  # image is flipped top-bottom for some reason

        glfw.swap_buffers(self.window)

        if save_image:
            logger().debug("Save image...")
            image.save(name)

        if return_img:
            return image

    def should_close(self):
        return glfw.window_should_close(self.window)

    @staticmethod
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
