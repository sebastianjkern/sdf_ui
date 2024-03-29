__docformat__ = "google"

import ttfquery
from ttfquery import describe, glyph

from .core import line, bezier
from .util import collinear

def middle(x1, y1, x2, y2):
    """
    Calculate the midpoint between two points.

    Args:
    - x1, y1: Coordinates of the first point.
    - x2, y2: Coordinates of the second point.

    Returns:
    Tuple: Midpoint coordinates (x, y).
    """
    return x1 + 0.5 * (x2 - x1), y1 + 0.5 * (y2 - y1)


def get_glyph(font_file_path, char):
    """
    Extract the Bezier control points of a glyph from a font file.

    Args:
    - font_file_path (str): Path to the font file.
    - char (str): Character for which to extract the glyph.

    Returns:
    List: List of Bezier control points for the glyph.
    """
    font = describe.openFont(font_file_path)
    g = glyph.Glyph(ttfquery.glyphquery.glyphName(font, char))
    ttf_contours = g.calculateContours(font)


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


def glyph_sdf(ctx, glyph, scale, ox, oy, path="fonts/SFUIDisplay-Bold.ttf"):
    """
    Generate the signed distance field (SDF) for a glyph.

    Parameters:
    - glyph (str): The character for which to generate the SDF.
    - scale (float): Scaling factor for the glyph.
    - ox, oy (float): Offset of the glyph.
    - path (str): Path to the font file.

    Returns:
    SDFTexture: The union of Bezier curves representing the SDF.
    """
    control_points = get_glyph(path, glyph)

    union_sdf = None

    for ix, shape in enumerate(control_points):
        for iy, stroke in enumerate(shape):
            a = (stroke[0][0] * scale + ox, stroke[0][1] * scale + oy)
            b = (stroke[1][0] * scale + ox, stroke[1][1] * scale + oy)
            c = (stroke[2][0] * scale + ox, stroke[2][1] * scale + oy)

            if not collinear(*a, *b, *c):
                bezier_sdf = bezier(ctx, a, b, c)
            else:
                bezier_sdf = line(ctx, a, c)

            if iy == 0 and ix == 0:
                union_sdf = bezier_sdf
            else:
                union_sdf = union_sdf.union(bezier_sdf)

    return union_sdf
