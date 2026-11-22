import math
from types import SimpleNamespace

import pytest

from sdf_ui.export import PlotterProfile, write_gcode, write_svg
from sdf_ui.vector import VectorDrawing, VectorPath


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
        "G17",
        "G94",
        "M5",
        "T1",
        "M5",
        "G0 X4 Y14 F1200",
        "M3",
        "G1 X16 Y6 F600",
        "M5",
        "M2",
    ]


def test_write_gcode_allows_custom_or_omitted_program_end(tmp_path):
    destination = tmp_path / "drawing.gcode"

    write_gcode(
        _drawing(pen=None), destination, plotter=_profile(program_end_command="M30")
    )

    assert destination.read_text().splitlines()[-1] == "M30"
    write_gcode(
        _drawing(pen=None), destination, plotter=_profile(program_end_command=None)
    )
    assert destination.read_text().splitlines()[-1] == "M5"


def test_write_gcode_optimizes_path_order_and_orientation_within_pen_run(tmp_path):
    drawing = SimpleNamespace(
        size=(20, 20),
        paths=(
            SimpleNamespace(
                points=((15, 15), (18, 15)), color="a", width=1, pen="black"
            ),
            SimpleNamespace(points=((1, 1), (2, 1)), color="b", width=1, pen=None),
            SimpleNamespace(points=((5, 1), (6, 1)), color="c", width=1, pen=None),
        ),
    )
    destination = tmp_path / "optimized.gcode"

    write_gcode(
        drawing,
        destination,
        plotter=_profile(size_mm=(40, 40), optimize_travel=True),
    )

    lines = destination.read_text().splitlines()
    travels = [line for line in lines if line.startswith("G0 ")]
    assert travels[:3] == [
        "G0 X30 Y10 F1200",
        "G0 X12 Y38 F1200",
        "G0 X4 Y38 F1200",
    ]
    assert lines.index("T1") < lines.index(travels[0])
    assert lines[-1] == "M2"


def test_write_gcode_optimization_preserves_pen_run_order(tmp_path):
    drawing = VectorDrawing(
        (10, 10),
        (
            VectorPath(((1, 1), (2, 1)), "a", 1, "black"),
            VectorPath(((3, 1), (4, 1)), "b", 1),
            VectorPath(((8, 8), (9, 8)), "c", 1, "red"),
            VectorPath(((5, 8), (6, 8)), "d", 1),
            VectorPath(((1, 8), (2, 8)), "e", 1, "black"),
        ),
    )
    destination = tmp_path / "pen-runs.gcode"
    profile = _profile(
        optimize_travel=True,
        pen_change_commands={"black": "T1", "red": "T2"},
    )

    write_gcode(drawing, destination, plotter=profile)

    lines = destination.read_text().splitlines()
    travels = [index for index, line in enumerate(lines) if line.startswith("G0 ")]
    assert [line for line in lines if line.startswith("T")] == ["T1", "T2", "T1"]
    tool_changes = [index for index, line in enumerate(lines) if line.startswith("T")]
    assert tool_changes[0] < travels[0] < tool_changes[1] < travels[2]
    assert travels[2] < tool_changes[2] < travels[4]


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
