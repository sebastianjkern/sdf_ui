"""Render a seeded Alpine engraving as SVG, G-code, and a matching preview."""

import argparse
import math
import random
from itertools import groupby
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from sdf_ui.export import PlotterProfile, write_gcode, write_svg
from sdf_ui.fonts import default_font_path
from sdf_ui.vector import VectorDrawing, VectorPath, contour_lines

OUTPUT = Path(__file__).resolve().parents[1] / "out" / "examples" / "alpine_engraving"
SEED = 31415
SIZE = (1200, 800)
TEXT_SCALE = 6
PAPER = "#f7f2e7"
INK = "#263844"
PALE_INK = "#7a8990"
SLATE = "#465a65"
DISTANT = "#9ba4a5"


def _ridge(seed, anchors, jitter=5.0):
    rng = random.Random(seed)
    points = []
    for index, (x0, y0) in enumerate(anchors[:-1]):
        x1, y1 = anchors[index + 1]
        steps = max(8, int((x1 - x0) / 9))
        for step in range(steps):
            t = step / steps
            smooth = t * t * (3 - 2 * t)
            points.append(
                (
                    x0 + (x1 - x0) * t,
                    y0 + (y1 - y0) * smooth + rng.uniform(-jitter, jitter),
                )
            )
    points.append(anchors[-1])
    return points


def _inside(point, polygon):
    x, y = point
    result = False
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = previous
        x1, y1 = current
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            result = not result
        previous = current
    return result


def _hatch(polygon, exclusions, *, angle, spacing, seed, color, width, strength):
    rng = random.Random(seed)
    direction = (math.cos(angle), math.sin(angle))
    normal = (-direction[1], direction[0])
    projections = [x * normal[0] + y * normal[1] for x, y in polygon]
    offset = min(projections) + rng.uniform(0, spacing)
    high = max(projections)
    while offset < high:

        def point(along, bow=0):
            return (
                direction[0] * along + normal[0] * (offset + bow),
                direction[1] * along + normal[1] * (offset + bow),
            )

        crossings = sorted(_line_intersections(polygon, normal, direction, offset))
        occlusions = [
            value
            for shape in exclusions
            for value in _line_intersections(shape, normal, direction, offset)
        ]
        for start, end in zip(crossings[::2], crossings[1::2]):
            cuts = sorted([start, end, *(v for v in occlusions if start < v < end)])
            for left, right in zip(cuts, cuts[1:]):
                if any(
                    _inside(point((left + right) * 0.5), shape) for shape in exclusions
                ):
                    continue
                cursor = left + rng.uniform(0.7, 3.8)
                while cursor < right - 3:
                    tip = min(right - 0.7, cursor + rng.uniform(9, 52))
                    center = (cursor + tip) * 0.5
                    x, y = point(center)
                    density = _ink_density(polygon, x, y, strength)
                    if rng.random() < density:
                        curved = point(center, rng.uniform(-1.25, 1.25))
                        if not _inside(curved, polygon) or any(
                            _inside(curved, shape) for shape in exclusions
                        ):
                            curved = (x, y)
                        points = (point(cursor), curved, point(tip))
                        yield VectorPath(points, color, width * rng.uniform(0.73, 1.24))
                    cursor = tip + rng.uniform(1.3, 10.5)
        offset += spacing * rng.uniform(0.77, 1.23)


def _ink_density(polygon, x, y, strength):
    skyline = _ridge_y(polygon, x)
    rise = (_ridge_y(polygon, x + 9) - _ridge_y(polygon, x - 9)) / 18
    extent = max(30, polygon[-1][1] - skyline)
    height = min(1, max(0, (y - skyline) / extent))
    ribs = math.sin(x * 0.045 + y * 0.079 + math.sin(x * 0.033) * 1.5)
    grain = math.sin(x * 0.093 - y * 0.14)
    shadow = 0.48 + 0.18 * math.tanh(rise) + 0.24 * ribs + 0.10 * grain
    edge = min(1, (4 + min(x, SIZE[0] - x)) / 24)
    return min(1, strength * shadow * (1 - height**2.3) * edge)


def _ridge_y(polygon, x):
    x = min(SIZE[0], max(0, x))
    for first, second in zip(polygon, polygon[1:]):
        if second[0] > first[0] and first[0] <= x <= second[0]:
            ratio = (x - first[0]) / (second[0] - first[0])
            return first[1] + (second[1] - first[1]) * ratio
    return 0.0


def _line_intersections(polygon, normal, direction, offset):
    values = []
    previous = polygon[-1]
    for current in polygon:
        a = previous[0] * normal[0] + previous[1] * normal[1] - offset
        b = current[0] * normal[0] + current[1] * normal[1] - offset
        if (a <= 0 < b) or (b <= 0 < a):
            t = a / (a - b)
            x = previous[0] + t * (current[0] - previous[0])
            y = previous[1] + t * (current[1] - previous[1])
            values.append(x * direction[0] + y * direction[1])
        previous = current
    return values


def _mountain_layer(ridge, bottom):
    return [*ridge, (SIZE[0], bottom), (0, bottom)]


def _make_drawing(seed):
    distant = _ridge(
        seed + 4,
        [
            (0, 365),
            (72, 329),
            (145, 353),
            (241, 274),
            (310, 322),
            (386, 293),
            (455, 330),
            (525, 252),
            (596, 316),
            (686, 281),
            (753, 330),
            (852, 260),
            (926, 323),
            (1018, 285),
            (1100, 335),
            (1200, 302),
        ],
        4,
    )
    middle = _ridge(
        seed + 8,
        [
            (0, 455),
            (96, 403),
            (174, 428),
            (254, 368),
            (330, 408),
            (410, 334),
            (475, 379),
            (538, 306),
            (592, 348),
            (658, 176),
            (711, 329),
            (785, 378),
            (877, 310),
            (956, 362),
            (1038, 321),
            (1124, 374),
            (1200, 350),
        ],
        5,
    )
    foreground = _ridge(
        seed + 12,
        [
            (0, 540),
            (93, 491),
            (174, 518),
            (261, 463),
            (338, 493),
            (420, 421),
            (489, 460),
            (559, 397),
            (624, 427),
            (692, 338),
            (742, 391),
            (818, 359),
            (898, 437),
            (988, 371),
            (1066, 426),
            (1142, 385),
            (1200, 403),
        ],
        4,
    )
    far_shape = _mountain_layer(distant, 800)
    mid_shape = _mountain_layer(middle, 800)
    front_shape = _mountain_layer(foreground, 800)
    paths = []
    central_snow = [
        (607, 290),
        (630, 236),
        (650, 199),
        (659, 176),
        (666, 193),
        (682, 233),
        (709, 296),
        (693, 285),
        (678, 267),
        (665, 282),
        (650, 260),
        (635, 282),
    ]
    for ridge, exclusions, shade, bottom, angle, spacing, strength in (
        (far_shape, (mid_shape, front_shape), DISTANT, 486, -0.62, 5.0, 0.85),
        (mid_shape, (front_shape, central_snow), SLATE, 561, 0.91, 3.6, 1.25),
        (front_shape, (), INK, 650, -0.72, 3.25, 1.42),
    ):
        polygon = ridge[:-2] + [(SIZE[0], bottom), (0, bottom)]
        paths.extend(
            _hatch(
                polygon,
                exclusions,
                angle=angle,
                spacing=spacing,
                seed=seed + bottom,
                color=shade,
                width=0.72 if shade == INK else 0.58,
                strength=strength,
            )
        )
        paths.extend(
            _hatch(
                polygon,
                exclusions,
                angle=angle + 1.04,
                spacing=spacing * 1.7,
                seed=seed - bottom,
                color=shade,
                width=0.54,
                strength=strength * 0.64,
            )
        )
        for index, point in enumerate(ridge[:-3]):
            next_point = ridge[index + 1]
            center = tuple((a + b) / 2 for a, b in zip(point, next_point))
            if all(
                not _inside(center, shape)
                for shape in exclusions
                if shape is not central_snow
            ):
                paths.append(
                    VectorPath(
                        (point, next_point), shade, 1.65 if bottom > 500 else 1.05
                    )
                )

    snow_edges = [
        central_snow[:7],
        [(510, 336), (526, 310), (538, 306), (551, 326), (576, 349)],
        [(219, 386), (241, 274), (257, 292), (284, 335)],
        [(840, 295), (852, 260), (867, 281), (891, 312)],
        [(395, 383), (410, 334), (427, 346), (449, 375)],
    ]
    rng = random.Random(seed)
    for edge in snow_edges:
        paths.append(VectorPath(tuple(edge), SLATE, 0.8))
        for index in range(5):
            t = index * 0.13 + rng.uniform(0.18, 0.25)
            first, last = edge[len(edge) // 2], edge[-1]
            x = first[0] + (last[0] - first[0]) * t
            y = first[1] + (last[1] - first[1]) * t
            tip = (x + rng.uniform(-12, -5), y + rng.uniform(9, 16))
            paths.append(VectorPath(((x, y), tip), PALE_INK, 0.52))

    paths = [
        VectorPath(
            tuple((52 + x * 1096 / SIZE[0], y) for x, y in path.points),
            path.color,
            path.width,
        )
        for path in paths
    ]
    paths.extend(
        (
            VectorPath(((52, 148), (1148, 148)), PALE_INK, 0.6),
            VectorPath(((1101, 126), (1128, 126), (1121, 122)), PALE_INK, 0.75),
            VectorPath(((1128, 126), (1121, 130)), PALE_INK, 0.75),
            VectorPath(((52, 704), (1148, 704)), PALE_INK, 0.6),
        )
    )
    return VectorDrawing(SIZE, tuple(paths))


def _lettering():
    font_path = default_font_path()
    title = ImageFont.truetype(font_path, size=34 * TEXT_SCALE)
    label = ImageFont.truetype(font_path, size=14 * TEXT_SCALE)
    for value, font, (x, y), color in (
        ("LES AIGUILLES", title, (52, 67), INK),
        ("ÉTUDE DES ALPES FRANÇAISES", label, (54, 114), SLATE),
        ("LUMIÈRE RASANTE", label, (1007, 97), SLATE),
        ("NEIGE  /  ROCHE  /  OMBRE", label, (53, 726), SLATE),
    ):
        left, top, right, bottom = font.getbbox(value)
        width = math.ceil((right - left) / TEXT_SCALE) + 2
        height = math.ceil((bottom - top) / TEXT_SCALE) + 2
        mask = Image.new("L", (width * TEXT_SCALE, height * TEXT_SCALE))
        ImageDraw.Draw(mask).text(
            (TEXT_SCALE - left, TEXT_SCALE - top), value, font=font, fill=255
        )
        yield mask, (x - 1, y - 1), color


def _plot_letters(lettering):
    row_step = 5
    for mask, (x, y), color in lettering:
        pixels = mask.load()
        for row in range(row_step // 2, mask.height, row_step):
            covered = groupby(
                range(mask.width), lambda column: pixels[column, row] >= 128
            )
            for is_ink, columns in covered:
                if not is_ink:
                    continue
                first, *rest = columns
                last = rest[-1] if rest else first
                baseline = y + (row + 0.5) / TEXT_SCALE
                yield VectorPath(
                    (
                        (x + first / TEXT_SCALE, baseline),
                        (x + (last + 1) / TEXT_SCALE, baseline),
                    ),
                    color,
                    row_step / TEXT_SCALE,
                )


def _save_svg(drawing, lettering, path):
    write_svg(drawing, path)
    shapes = []
    for mask, (x, y), color in lettering:
        rings = contour_lines(mask, level=127.5)
        commands = (
            "M "
            + " L ".join(
                f"{x + (u + 0.5) / TEXT_SCALE:.3f},{y + (v + 0.5) / TEXT_SCALE:.3f}"
                for u, v in ring[:-1]
            )
            + " Z"
            for ring in rings
        )
        shapes.append(
            f'<path fill="{color}" fill-rule="evenodd" d="{" ".join(commands)}"/>'
        )
    svg = path.read_text(encoding="utf-8")
    path.write_text(
        svg.removesuffix("</svg>\n") + "\n".join(shapes) + "\n</svg>\n",
        encoding="utf-8",
    )


def _save_png(drawing, lettering, path):
    scale = 3
    image = Image.new("RGB", (drawing.size[0] * scale, drawing.size[1] * scale), PAPER)
    draw = ImageDraw.Draw(image)
    for path_data in drawing.paths:
        scaled = [(x * scale, y * scale) for x, y in path_data.points]
        draw.line(
            scaled,
            fill=path_data.color,
            width=max(1, round(path_data.width * scale)),
            joint="curve",
        )
    preview = image.resize(drawing.size, Image.Resampling.LANCZOS)
    for mask, origin, color in lettering:
        alpha = mask.resize(
            (mask.width // TEXT_SCALE, mask.height // TEXT_SCALE),
            Image.Resampling.LANCZOS,
        )
        preview.paste(color, origin, alpha)
    preview.save(path)


def alpine_engraving(output=OUTPUT, seed=SEED):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    drawing = _make_drawing(seed)
    lettering = tuple(_lettering())
    profile = PlotterProfile(
        (240, 160), 0.2, "G0 Z5", "G0 Z0", 2600, 900, optimize_travel=True
    )
    _save_svg(drawing, lettering, output / "alpine_engraving.svg")
    plot = VectorDrawing(drawing.size, drawing.paths + tuple(_plot_letters(lettering)))
    write_gcode(plot, output / "alpine_engraving.gcode", plotter=profile)
    _save_png(drawing, lettering, output / "alpine_engraving.png")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(alpine_engraving(args.output, args.seed))
