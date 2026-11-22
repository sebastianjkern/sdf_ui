from xml.etree import ElementTree

import numpy as np
from PIL import Image

from sdf_ui import Canvas, paint, param, sdf
from sdf_ui.export import PlotterProfile

_VIEWPORT = (96, 88)
_CENTER = (48, 44)
_HOLE = 10
_COORDINATE_TOLERANCE = 1e-6


def _curved_scene():
    shape = sdf.circle(_CENTER, 33).subtract(sdf.circle(_CENTER, _HOLE))
    hatch = paint.HatchStyle(
        (paint.HatchPass(0, 7, phase=2),), hand=paint.HandStroke(bow=6, seed=8)
    )
    return paint.scene(paint.fill(shape, "#263b48", hatch=hatch))


def _svg_points(path):
    root = ElementTree.parse(path).getroot()
    return tuple(
        tuple(
            tuple(map(float, xy.split(","))) for xy in element.attrib["points"].split()
        )
        for element in root
    )


def _gcode_strokes(path):
    strokes = []
    for line in path.read_text().splitlines():
        if line.startswith("G0 X"):
            strokes.append([])
        if line.startswith(("G0 X", "G1 X")):
            _, x, y, _ = line.split()
            strokes[-1].append((float(x[1:]), _VIEWPORT[1] - float(y[1:])))
    return strokes


def test_bowed_strokes_reach_all_outputs_and_respect_the_cutout(tmp_path):
    scene = _curved_scene()
    plotter = PlotterProfile(_VIEWPORT, 1, "G0 Z3", "G0 Z0", 2000, 800)
    with Canvas(_VIEWPORT, device_scale=2) as canvas:
        drawing = scene.to_vector(canvas=canvas)
        svg = scene.save_svg(tmp_path / "curved.svg", canvas=canvas)
        gcode = scene.save_gcode(
            tmp_path / "curved.gcode", canvas=canvas, plotter=plotter
        )
        png = scene.save_png(tmp_path / "curved.png", canvas=canvas)
    assert drawing.size == _VIEWPORT
    svg_paths, gcode_paths = _svg_points(svg), _gcode_strokes(gcode)
    assert len(drawing.paths) == len(svg_paths) == len(gcode_paths)
    assert any(_is_bowed(path.points) for path in drawing.paths)
    for path, svg_path, gcode_path in zip(drawing.paths, svg_paths, gcode_paths):
        np.testing.assert_allclose(
            svg_path, path.points, atol=_COORDINATE_TOLERANCE, rtol=0
        )
        np.testing.assert_allclose(
            gcode_path, path.points, atol=_COORDINATE_TOLERANCE, rtol=0
        )
    _assert_ink_follows_paths(drawing, png)


def _assert_ink_follows_paths(drawing, png):
    with Image.open(png) as rendered:
        alpha = np.asarray(rendered)[..., 3]
    assert alpha[88, 96] == 0
    assert np.count_nonzero(alpha) > 300
    centers = np.array([path.points[len(path.points) // 2] for path in drawing.paths])
    x, y = (centers * 2).astype(int).T
    assert np.all(alpha[y, x] > 0)
    points = np.concatenate([path.points for path in drawing.paths])
    hole_center = (48.25, 43.75)
    assert np.min(np.linalg.norm(points - hole_center, axis=1)) > _HOLE - 1


def _is_bowed(points):
    if len(points) < 3:
        return False
    vertices = np.asarray(points)
    x, y = vertices[-1] - vertices[0]
    middle = vertices[len(vertices) // 2] - vertices[0]
    return abs(x * middle[1] - y * middle[0]) > 1


def _lit_scene():
    passes = (
        paint.HatchPass(0, 7, min_shadow=0.24),
        paint.HatchPass(np.pi / 2, 6, min_shadow=0.62),
    )
    light = paint.HatchLight((param("light_x"), 0, 0.6), depth=33)
    style = paint.HatchStyle(passes, light=light)
    return paint.scene(paint.fill(sdf.circle(_CENTER, 33), "#1d3349", hatch=style))


def test_reversing_light_moves_the_dense_hatch_to_the_other_side(tmp_path):
    drawing = _lit_scene()
    with Canvas(_VIEWPORT) as canvas:
        left = drawing.save_png(
            tmp_path / "left.png", canvas=canvas, params={"light_x": -1}
        )
        right = drawing.save_png(
            tmp_path / "right.png", canvas=canvas, params={"light_x": 1}
        )
        left_paths = drawing.to_vector(canvas=canvas, params={"light_x": -1}).paths
    assert _half_density(left)[0] < _half_density(left)[1]
    assert _half_density(right)[0] > _half_density(right)[1]
    vertical = [
        path
        for path in left_paths
        if abs(path.points[-1][0] - path.points[0][0]) < _COORDINATE_TOLERANCE
    ]
    lengths = np.array(
        [abs(path.points[-1][1] - path.points[0][1]) for path in vertical]
    )
    shadow_side = np.array(
        [path.points[0][0] > _CENTER[0] for path in vertical], dtype=bool
    )
    assert lengths[shadow_side].sum() > 5 * lengths[~shadow_side].sum()


def _half_density(path):
    with Image.open(path) as image:
        alpha = np.asarray(image)[..., 3]
    half = _CENTER[0]
    return alpha[:, :half].sum(), alpha[:, half:].sum()
