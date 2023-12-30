import ttfquery
from ttfquery import describe, glyph

from sdf_ui.core.core import line, bezier
from sdf_ui.core.util import collinear


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


def glyph_sdf(glyph, scale, ox, oy, path="fonts/SFUIDisplay-Bold.ttf"):
    control_points = get_glyph(path, glyph)

    union_sdf = None

    for ix, shape in enumerate(control_points):
        for iy, stroke in enumerate(shape):
            a = (stroke[0][0] * scale + ox, stroke[0][1] * scale + oy)
            b = (stroke[1][0] * scale + ox, stroke[1][1] * scale + oy)
            c = (stroke[2][0] * scale + ox, stroke[2][1] * scale + oy)

            if not collinear(*a, *b, *c):
                bezier_sdf = bezier(a, b, c)
            else:
                bezier_sdf = line(a, c)

            if iy == 0 and ix == 0:
                union_sdf = bezier_sdf
            else:
                union_sdf = union_sdf.union(bezier_sdf)

    return union_sdf
