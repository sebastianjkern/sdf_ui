import glfw

from sdf_ui.shader import Rect, Circle
from sdf_ui.util import hex_col
from sdf_ui.window import SdfUiContext

with SdfUiContext(500, 500, resizable=True, title="Render Engine") as context:
    col1 = hex_col("#e9c46a")
    col2 = hex_col("#2a9d8f")
    col3 = hex_col("#e76f51")
    col4 = hex_col("#550f96")
    white = hex_col("#ffffff")
    black = hex_col("#000000")
    bgr = hex_col("#2C2D35")

    rect = Rect()
    rect.elevation((.000010,))
    rect.center((0.2, 0))
    rect.size((0.60, 0.60))
    rect.corner_radius((0.03, 0.03, 0.03, 0.03))
    rect.color(col1)

    circle1 = Circle()
    circle1.elevation((.000010,))
    circle1.center((-.4, 0.1))
    circle1.radius((0.4,))
    circle1.color(col3)

    circle2 = Circle()
    circle2.elevation((.000010,))
    circle2.center((-.7, 0.2))
    circle2.radius((0.3,))
    circle2.color(col2)

    context.set_background(bgr)

    while not context.should_close():
        objects = [rect, circle1, circle2]

        start = glfw.get_time()
        context.render(objects)
        context.set_title(f"{glfw.get_time() - start}")
