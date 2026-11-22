import math
from xml.etree import ElementTree

from sdf_ui import Canvas, paint, param, sdf
from sdf_ui.export import PlotterProfile


def _hatch_scene():
    shape = sdf.circle((26, 10), 16).subtract(sdf.circle((26, 10), 4))
    style = paint.HatchStyle(
        (paint.HatchPass(0, 6, 1.6, param("phase")), paint.HatchPass(math.pi / 2, 80))
    )
    return paint.scene(paint.fill(shape, "#326a82", hatch=style))


def test_raster_and_vector_hatch_share_coordinates_and_cutout():
    scene = _hatch_scene()
    params = {"phase": 4.0}
    hole_x, hatch_y = 26.5, 29.5
    with Canvas((52, 40)) as canvas:
        with canvas.render(scene.to_color(), params) as rendered:
            pixels = rendered.read()
        vector = scene.to_vector(canvas=canvas, params=params)
        shifted = scene.to_vector(canvas=canvas, params={"phase": 3.0})
    assert pixels[10, 18, 3] > pixels[13, 18, 3]
    assert pixels[10, 26, 3] == 0
    horizontal = [path for path in vector.paths if path.points[0][1] == hatch_y]
    assert len(horizontal) == 2
    for path in horizontal:
        x = [point[0] for point in path.points]
        assert not min(x) < hole_x < max(x)
    assert shifted != vector


def test_paint_scene_saves_all_three_formats(tmp_path):
    scene = _hatch_scene()
    params = {"phase": 4.0}
    profile = PlotterProfile((52, 40), 1, "G0 Z3", "G0 Z0", 2000, 800)
    with Canvas((52, 40)) as canvas:
        vector = scene.to_vector(canvas=canvas, params=params)
        svg = scene.save_svg(tmp_path / "drawing.svg", canvas=canvas, params=params)
        png = scene.save_png(tmp_path / "drawing.png", canvas=canvas, params=params)
        gcode = scene.save_gcode(
            tmp_path / "drawing.gcode", canvas=canvas, params=params, plotter=profile
        )
    assert png.stat().st_size > 0
    assert len(ElementTree.parse(svg).getroot()) == len(vector.paths)
    assert gcode.read_text().count("G0 Z0\n") == len(vector.paths)


def test_outline_and_isolines_share_the_original_sdf():
    shape = sdf.circle((16, 16), 8)
    scene = paint.scene(
        paint.outline(shape, "#303030"),
        paint.isolines(shape, "#0090a0", spacing=6),
    )
    with Canvas((32, 32)) as canvas:
        vector = scene.to_vector(canvas=canvas)
    assert vector.paths[0].color == "#303030"
    assert vector.paths[0].points[0] == vector.paths[0].points[-1]
    assert sum(path.color == "#0090a0" for path in vector.paths) > 1
