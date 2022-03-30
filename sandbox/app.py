import numpy as np

from sdf_ui.window import *


def rgb_col(r: int, g: int, b: int, *a):
    values = np.array([r, g, b, *a], dtype=np.float)
    values /= 255
    return tuple(values.tolist())


def hex_col(string: str):
    g = string.lstrip("#")
    col = tuple(int(g[i:i + 2], 16) for i in (0, 2, 4))
    return rgb_col(*col, 255)


if __name__ == '__main__':
    width, height = 1280, 720

    window = init_glfw(width, height, visible=True)

    col1 = hex_col("#4E7FBA")
    col2 = hex_col("#76A9AE")
    col3 = hex_col("#C69463")
    col4 = hex_col("#2C2D35")

    bgr = ObjectDescriptor(RECT)
    bgr.uniforms.COLOR[1] = (1.0, 1.0, 1.0, 1.0)
    bgr.uniforms.CENTER[1] = (0.5, 0.5)
    bgr.uniforms.SIZE[1] = (0.5 * width / height, 1.0)

    obj1 = ObjectDescriptor(RECT)

    obj1.uniforms.ELEVATION[1] = (.000010,)
    obj1.uniforms.CENTER[1] = (0.6, 0.5)
    obj1.uniforms.SIZE[1] = (0.30, 0.30)
    obj1.uniforms.CORNER_RADIUS[1] = (0.03, 0.03, 0.03, 0.03)
    obj1.uniforms.COLOR[1] = col1

    obj2 = ObjectDescriptor(CIRCLE)

    obj2.uniforms.ELEVATION[1] = (.000010,)
    obj2.uniforms.CENTER[1] = (0.30, 0.55)
    obj2.uniforms.RADIUS[1] = (0.2,)
    obj2.uniforms.COLOR[1] = col3

    obj3 = ObjectDescriptor(CIRCLE)

    obj3.uniforms.ELEVATION[1] = (.000010,)
    obj3.uniforms.CENTER[1] = (0.15, 0.60)
    obj3.uniforms.RADIUS[1] = (0.15,)
    obj3.uniforms.COLOR[1] = col2

    # while not glfw.window_should_close(window):
    render(window, [bgr, obj1, obj2, obj3], save_image=True)

    ShaderProgram().cleanup()

    terminate_glfw(window)
