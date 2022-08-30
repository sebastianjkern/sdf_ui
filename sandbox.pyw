from sdf_ui.util import hex_col
from sdf_ui.window import SdfUiContext, Rect, Circle, Line

with SdfUiContext(720, int(720 / 16 * 10)) as context:
    col1 = hex_col("#e9c46a")
    col2 = hex_col("#2a9d8f")
    col3 = hex_col("#e76f51")
    col4 = hex_col("#2C2D35")
    col5 = hex_col("#550f96")

    bgr = Rect()
    bgr.color(col4)
    bgr.center((0.5, 0.5))
    bgr.size((2.0, 2.0))

    rect = Rect()
    rect.elevation((.000010,))
    rect.center((0.6, 0.5))
    rect.size((0.30, 0.30))
    rect.corner_radius((0.03, 0.03, 0.03, 0.03))
    rect.color(col1)

    circle1 = Circle()
    circle1.elevation((.000010,))
    circle1.center((0.30, 0.55))
    circle1.radius((0.2,))
    circle1.color(col3)

    circle2 = Circle()
    circle2.elevation((.000010,))
    circle2.center((0.15, 0.60))
    circle2.radius((0.15,))
    circle2.color(col2)

    line = Line()
    line.elevation((.0000010,))
    line.radius(.001)
    line.a((0.5, 0.5,))
    line.b((0.6, 0.7,))
    line.color(col5)

    objects = [bgr, rect, circle1, circle2, line]

    while not context.should_close():
        context.render(objects)
