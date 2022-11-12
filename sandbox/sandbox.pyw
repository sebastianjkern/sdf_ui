import glfw
import ttfquery
import ttfquery.glyph as glyph
from ttfquery import describe, glyphquery

from sdf_ui.shader import Rect, Circle, Bezier, Line
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


    def get_glyph(font_file_path, char):
        font = describe.openFont(font_file_path)
        g = glyph.Glyph(ttfquery.glyphquery.glyphName(font, char))
        ttf_contours = g.calculateContours(font)

        def middle(x1, y1, x2, y2):
            return x1 + 0.5 * (x2 - x1), y1 + 0.5 * (y2 - y1)

        interpolated_contours = []

        for contour in ttf_contours:
            interpolated_points = [list(contour[0][0])]

            for i in range(1, len(contour)):
                if contour[i][1] == contour[i - 1][1]:
                    interpolated_points.append(
                        [*middle(
                            contour[i - 1][0][0],
                            contour[i - 1][0][1],
                            contour[i][0][0],
                            contour[i][0][1])])
                interpolated_points.append(list(contour[i][0]))

            interpolated_contours.append(interpolated_points)

        bezier_control_points = []

        for contour in interpolated_contours:
            bezier_curves_control_points = []

            for i in range(0, len(contour) - 1, 2):
                bezier_curves_control_points.append(contour[0 + i:3 + i])

            bezier_control_points.append(bezier_curves_control_points)

        return bezier_control_points


    # TODO: Stencil fill? Trough polygons ->  different than the aim of the rendering technique

    render_objects = []

    def collinear(x1, y1, x2, y2, x3, y3):
        area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        return area <= 0.000000001


    scale = 1 / 2000
    offx, offy = 0.0, -0.25

    contours = get_glyph("georgia_regular.ttf", "%")

    for section in contours:
        for stroke in section:
            ax, ay = stroke[0][0] * scale + offx, stroke[0][1] * scale + offy
            bx, by = stroke[1][0] * scale + offx, stroke[1][1] * scale + offy
            cx, cy = stroke[2][0] * scale + offx, stroke[2][1] * scale + offy

            if not collinear(ax, ay, bx, by, cx, cy):
                bezier = Bezier()
                bezier.a((ax, ay))
                bezier.b((bx, by))
                bezier.c((cx, cy))
                # bezier.radius(0.01)
                bezier.color(hex_col("#111111"))

                render_objects.append(bezier)

            else:
                line = Line()
                line.a((ax, ay))
                line.b((cx, cy))
                line.radius(0.0001)
                line.color(hex_col("#111111"))

                render_objects.append(line)

    context.background = hex_col("#ffffff")

    for index, obj in enumerate(render_objects):
        objects = [obj]

        context.render(objects, save_image=False, image_name=f"{str(index).zfill(4)}.png")

    while not context.should_close():
        objects = render_objects

        start = glfw.get_time()
        context.render(objects)
        context.set_title(f"{glfw.get_time() - start}")
