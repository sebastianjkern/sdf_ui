import math
from types import SimpleNamespace

import pytest

from sdf_ui.export import PlotterProfile, write_gcode, write_svg


def _drawing(points=((2, 3), (8, 7)), pen="black"):
    return SimpleNamespace(
        size=(10, 10),
        paths=(
            SimpleNamespace(
                points=points,
                color="#123456",
                width=0.5,
                pen=pen,
            ),
        ),
    )


def _profile(**overrides):
    values = {
        "size_mm": (20, 20),
        "mm_per_unit": 2,
        "pen_up_command": "M5",
        "pen_down_command": "M3",
        "travel_speed_mm_min": 1200,
        "draw_speed_mm_min": 600,
        "pen_change_commands": {"black": "T1"},
    }
    values.update(overrides)
    return PlotterProfile(**values)


def test_write_svg_preserves_canvas_coordinates_and_style(tmp_path):
    destination = tmp_path / "drawing.svg"

    result = write_svg(_drawing(), destination)

    assert result == destination
    assert 'viewBox="0 0 10 10"' in destination.read_text()
    assert 'points="2,3 8,7"' in destination.read_text()
    assert 'stroke="#123456"' in destination.read_text()


def test_write_gcode_scales_inverts_y_and_lifts_between_paths(tmp_path):
    drawing = _drawing()
    destination = tmp_path / "drawing.gcode"

    result = write_gcode(drawing, destination, plotter=_profile())

    lines = destination.read_text().splitlines()
    assert result == destination
    assert lines == [
        "G21",
        "G90",
        "M5",
        "T1",
        "M5",
        "G0 X4 Y14 F1200",
        "M3",
        "G1 X16 Y6 F600",
        "M5",
    ]


def test_write_gcode_rejects_coordinates_outside_workspace(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        write_gcode(
            _drawing(points=((11, 2),)),
            tmp_path / "drawing.gcode",
            plotter=_profile(),
        )


def test_write_gcode_rejects_drawing_larger_than_physical_workspace(tmp_path):
    with pytest.raises(ValueError, match="does not fit"):
        write_gcode(
            _drawing(),
            tmp_path / "drawing.gcode",
            plotter=_profile(size_mm=(10, 10)),
        )


def test_writers_snap_boundary_roundoff_to_canvas_edge(tmp_path):
    drawing = _drawing(
        points=(
            (math.nextafter(0.0, -math.inf), 2),
            (math.nextafter(10.0, math.inf), 7),
        ),
        pen=None,
    )

    svg = write_svg(drawing, tmp_path / "roundoff.svg").read_text()
    gcode = write_gcode(
        drawing,
        tmp_path / "roundoff.gcode",
        plotter=_profile(),
    ).read_text()

    assert 'points="0,2 10,7"' in svg
    assert "G0 X0 Y16 F1200" in gcode
    assert "G1 X20 Y6 F600" in gcode


def test_writers_reject_coordinates_far_outside_canvas(tmp_path):
    drawing = _drawing(points=((10 + 100 * math.ulp(10.0), 2),))

    with pytest.raises(ValueError, match="outside"):
        write_svg(drawing, tmp_path / "outside.svg")
