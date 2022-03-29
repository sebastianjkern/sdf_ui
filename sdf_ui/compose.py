from pathlib import Path

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
