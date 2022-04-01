from sdf_ui.window import ObjectDescriptor, init_glfw, render, terminate_glfw
from sdf_ui.shader import ShaderProgram, RECT, LINE, CIRCLE
from sdf_ui.colors import hex_col

import glfw

if __name__ == '__main__':
    width, height = 1080, int(1080 / 16 * 10)

    window = init_glfw(width, height, visible=True)

    col1 = hex_col("#e9c46a")
    col2 = hex_col("#2a9d8f")
    col3 = hex_col("#e76f51")
    col4 = hex_col("#2C2D35")

    bgr = ObjectDescriptor(RECT)
    bgr.uniforms.COLOR[1] = col4
    bgr.uniforms.CENTER[1] = (0.5, 0.5)
    bgr.uniforms.SIZE[1] = (2.0, 2.0)

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

    while not glfw.window_should_close(window):
        render(window, [bgr, obj1, obj2, obj3, ], save_image=False)

    ShaderProgram().cleanup()

    terminate_glfw(window)
