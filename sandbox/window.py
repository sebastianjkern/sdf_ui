import sys
import traceback
from typing import List

import glfw
from OpenGL.GL import *
from PIL import ImageOps, Image

from sdf_ui.shader import RenderObject, ShaderCache
from sdf_ui.util import logger, hex_col


class SdfUiContext:
    def __init__(self, width=500, height=500, resizable=True, visible=True, title="SDF GPU Renderer", vsync=True):
        self.width = width
        self.height = height
        self.resizable = resizable
        self.visible = visible
        self.window = None
        self._title = title

        self.shader_cache = ShaderCache()

        self.background = hex_col("#ffffff")

        self._last_time = 0
        self._now_time = 0
        self.dt = 0

        self.image = None

        self.vsync = vsync

    def set_title(self, title):
        self._title = title
        glfw.set_window_title(self.window, title)

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

        self.window = glfw.create_window(self.width, self.height, self._title, None, None)

        if not self.window:
            msg = ValueError("Failed to create glfw window").with_traceback(traceback.print_stack())
            logger().error(msg)
            glfw.terminate()
            sys.exit(-1)

        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if self.vsync else 0)

        logger().debug(glGetString(GL_VERSION).decode("UTF-8"))

        self._last_time = glfw.get_time()
        self._now_time = glfw.get_time()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for shader in self.shader_cache.cache:
            self.shader_cache.cache[shader].cleanup()

        glfw.destroy_window(self.window)
        glfw.terminate()

    def render(self, objects: List[RenderObject], save_image=False,
               return_img=False, image_name="glutout.png"):

        self._now_time = glfw.get_time()
        self.dt += (self._now_time - self._last_time) / (1 / 60)
        self._last_time = self._now_time

        width, height = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, width, height)

        glDisable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(*self.background)

        for obj in objects:
            if height != 0:
                obj.vertical_stretch((width / height,))
                obj.antialiasing_distance((4.0 / height,))
                obj.bb()

            shader = self.shader_cache.get_shader(obj.get_shader_name())
            shader.activate_and_draw(obj.get_uniforms())

        if save_image or return_img:
            data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
            self.image = Image.frombytes("RGBA", (width, height), data)
            self.image = ImageOps.flip(self.image)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

        if save_image:
            logger().debug("Save image...")
            self.image.save(image_name)

        if return_img:
            return self.image

    def should_close(self):
        return glfw.window_should_close(self.window)

    def px(self):
        width, height = glfw.get_framebuffer_size(self.window)
        m = min(width, height)
        return 1.0 / m