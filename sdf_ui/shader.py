from pathlib import Path

from OpenGL.GL import *

from .singleton import Singleton


def check_compile_status(shader):
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        log = glGetShaderInfoLog(shader)
        glDeleteShader(shader)
        print(log.decode())


def check_program_status(program):
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        log = glGetProgramInfoLog(program)
        glDeleteProgram(program)
        print(log.decode())


def compile_shader_program(
        vertex_source=None,
        fragment_source=None,
        geometry_source=None):
    if vertex_source is None:
        vs_path = Path(__file__).parent / "shader_files/vertex.glsl"
        vertex_source = open(vs_path).read()
    if fragment_source is None:
        fs_path = Path(__file__).parent / "shader_files/fragment.glsl"
        fragment_source = open(fs_path).read()
    if geometry_source is None:
        gs_path = Path(__file__).parent / "shader_files/geo.glsl"
        geometry_source = open(gs_path).read()

    vs = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vs, vertex_source)
    glCompileShader(vs)

    check_compile_status(vs)

    fs = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fs, fragment_source)
    glCompileShader(fs)

    check_compile_status(fs)

    gs = glCreateShader(GL_GEOMETRY_SHADER)
    glShaderSource(gs, geometry_source)
    glCompileShader(gs)

    check_compile_status(gs)

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vs)
    glAttachShader(shader_program, fs)
    glAttachShader(shader_program, gs)

    glLinkProgram(shader_program)
    glValidateProgram(shader_program)

    glDetachShader(shader_program, vs)
    glDetachShader(shader_program, fs)
    glDetachShader(shader_program, gs)

    glDeleteShader(vs)
    glDeleteShader(fs)
    glDeleteShader(gs)

    check_program_status(shader_program)

    return shader_program


RECT = "RECT"
LINE = "LINE"
CIRCLE = "CIRCLE"


def version(v: int) -> str:
    return f"#version {v} \r\n"


def define(name: str, val: str) -> str:
    return f'#define {name} {val} \r\n'


def fragment_shader(type_string: str):
    v = version(330)
    defines = define(type_string, "")

    fs_path = Path(__file__).parent / "shader_files/fragment.glsl"
    fragment_source = open(fs_path).read()

    return f'{v}{defines}{fragment_source}'.encode('UTF-8')


class ShaderProgram(Singleton):
    _rect = None
    _line = None
    _circle = None

    @property
    def rect(self):
        if ShaderProgram._rect is None:
            ShaderProgram._rect = compile_shader_program(fragment_source=fragment_shader(RECT))
        return ShaderProgram._rect

    @property
    def line(self):
        if ShaderProgram._line is None:
            ShaderProgram._line = compile_shader_program(fragment_source=fragment_shader(LINE))
        return ShaderProgram._line

    @property
    def circle(self):
        if ShaderProgram._circle is None:
            ShaderProgram._circle = compile_shader_program(fragment_source=fragment_shader(CIRCLE))
        return ShaderProgram._circle

    def cleanup(self):
        if ShaderProgram._rect is not None:
            glDeleteProgram(ShaderProgram._rect)
        if ShaderProgram._line is not None:
            glDeleteProgram(ShaderProgram._line)
        if ShaderProgram._circle is not None:
            glDeleteProgram(ShaderProgram._circle)
