import sys
from abc import abstractmethod
from pathlib import Path

from OpenGL.GL import *
from namedlist import namedlist

from .util import logger, Singleton


class ShaderTypes:
    RECT = "RECT"
    LINE = "LINE"
    CIRCLE = "CIRCLE"


class Shader:
    def __init__(self, shader_type) -> None:
        self.uniform_locations = dict()

        self.shader_type = shader_type

        self.program_id = self.compile_shader_program(fragment_source=self.fragment_shader(shader_type))

    @staticmethod
    def version(v: int) -> str:
        return f"#version {v} \r\n"

    @staticmethod
    def define(name: str, val: str) -> str:
        return f'#define {name} {val} \r\n'

    def fragment_shader(self, type_string: str):
        v = self.version(460)
        defines = self.define(type_string, "")

        fs_path = Path(__file__).parent / "shader/fragment.glsl"
        fragment_source = open(fs_path).read()

        return f'{v}{defines}{fragment_source}'.encode('UTF-8')

    @staticmethod
    def check_compile_status(shader):
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            log = glGetShaderInfoLog(shader)
            glDeleteShader(shader)
            logger().critical(log.decode("UTF-8"))
            sys.exit(1)

    @staticmethod
    def check_program_status(program):
        if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
            log = glGetProgramInfoLog(program)
            glDeleteProgram(program)
            logger().critical(log.decode("UTF-8"))
            sys.exit(1)

    def compile_shader_program(self,
                               vertex_source=None,
                               fragment_source=None,
                               geometry_source=None):
        logger().debug("Start shader program compilation...")

        if vertex_source is None:
            vs_path = Path(__file__).parent / "shader/vertex.glsl"
            vertex_source = open(vs_path).read()
        if fragment_source is None:
            fs_path = Path(__file__).parent / "shader/fragment.glsl"
            fragment_source = open(fs_path).read()
        if geometry_source is None:
            gs_path = Path(__file__).parent / "shader/geo.glsl"
            geometry_source = open(gs_path).read()

        vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vs, vertex_source)
        glCompileShader(vs)

        logger().debug("Compile vertex shader...")

        self.check_compile_status(vs)

        fs = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fs, fragment_source)
        glCompileShader(fs)

        logger().debug("Compile fragment shader...")

        self.check_compile_status(fs)

        gs = glCreateShader(GL_GEOMETRY_SHADER)
        glShaderSource(gs, geometry_source)
        glCompileShader(gs)

        logger().debug("Compile geometry shader...")

        self.check_compile_status(gs)

        shader_program = glCreateProgram()
        glAttachShader(shader_program, vs)
        glAttachShader(shader_program, fs)
        glAttachShader(shader_program, gs)

        logger().debug("Link shader program...")

        glLinkProgram(shader_program)
        glValidateProgram(shader_program)

        glDetachShader(shader_program, vs)
        glDetachShader(shader_program, fs)
        glDetachShader(shader_program, gs)

        glDeleteShader(vs)
        glDeleteShader(fs)
        glDeleteShader(gs)

        self.check_program_status(shader_program)

        return shader_program

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

    def activate_and_draw(self, uniforms):
        glUseProgram(self.program_id)

        for uniform_identifier in uniforms.__slots__:
            if uniform_identifier in self.uniform_locations:
                location = self.uniform_locations[uniform_identifier]
            else:
                location = glGetUniformLocation(self.program_id, uniform_identifier.encode("UTF-8"))
                self.uniform_locations[uniform_identifier] = location
            self.set_uniform(location, getattr(uniforms, uniform_identifier))

        glDrawArrays(GL_POINTS, 0, 1)
        glUseProgram(0)

    def cleanup(self):
        logger().debug(f"Delete {self.shader_type} shader program...")

        glDeleteProgram(self.program_id)


class ShaderCache:
    def __init__(self):
        self.cache = dict()

    def get_shader(self, shader_name: str):
        if shader_name in self.cache:
            return self.cache[shader_name]
        else:
            shader = Shader(shader_name)
            self.cache[shader_name] = shader
            return shader


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
    ("b", _t.VEC2),
    ("bb", _t.VEC4)])


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

    def bb(self):
        self._uniforms.bb = self.get_bounding_box()

    @abstractmethod
    def get_bounding_box(self) -> tuple:
        return -1, 1, 1, -1


class Rect(RenderObject):
    def __init__(self):
        super(Rect, self).__init__(ShaderTypes.RECT)

    def center(self, value):
        self._uniforms.center = value

    def size(self, value):
        self._uniforms.size = value

    def corner_radius(self, value):
        self._uniforms.corner_radius = value

    def get_bounding_box(self) -> tuple:
        c1, c2 = self._uniforms.center
        s1, s2 = self._uniforms.size

        min_x = c1 - s1 - 0.05
        max_x = c1 + s1 + 0.05

        min_y = c2 - s2 - 0.05
        max_y = c2 + s2 + 0.05

        return min_x, max_y, max_x, min_y


class Circle(RenderObject):
    def __init__(self):
        super(Circle, self).__init__(ShaderTypes.CIRCLE)

    def center(self, value):
        self._uniforms.center = value

    def radius(self, value):
        self._uniforms.radius = value

    def get_bounding_box(self) -> tuple:
        c1, c2 = self._uniforms.center
        r = self._uniforms.radius[0]

        min_x = c1 - r - 0.05
        max_x = c1 + r + 0.05

        min_y = c2 - r - 0.05
        max_y = c2 + r + 0.05

        return min_x, max_y, max_x, min_y


class Line(RenderObject):
    def __init__(self):
        super(Line, self).__init__(ShaderTypes.LINE)

    def a(self, value):
        self._uniforms.a = value

    def b(self, value):
        self._uniforms.b = value

    def radius(self, value):
        self._uniforms.radius = value

    def get_bounding_box(self) -> tuple:
        x1, y1 = self._uniforms.a
        x2, y2 = self._uniforms.b
        r = self._uniforms.radius

        min_x = min(x1, x2) - r - 0.05
        max_x = max(x1, x2) + r + 0.05

        min_y = min(y1, y2) - r - 0.05
        max_y = max(y1, y2) + r + 0.05

        return min_x, max_y, max_x, min_y
