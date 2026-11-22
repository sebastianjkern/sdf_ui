import numpy as np

from sdf_ui.vector import contour_lines, hatch_lines

EDGE_CONTOUR_X = 1.5


def test_hatches_preserve_hole_as_two_intervals():
    field = np.ones((7, 7))
    field[1:6, 1:6] = -1
    field[3, 3] = 1

    lines = hatch_lines(field, level=0, angle=0, spacing=1, phase=0)

    middle = [line for line in lines if line[0][1] == 3]
    assert len(middle) == 2
    assert middle[0][1][0] < 3
    assert middle[1][0][0] > 3


def test_hatch_clips_boundary_crossing_contour_to_half_pixel_edge():
    field = np.tile(np.array([-1.0, -1.0, 1.0, 1.0]), (4, 1))

    lines = hatch_lines(field, level=0, angle=0, spacing=1, phase=0)

    assert lines
    assert all(line[0][0] == -0.5 for line in lines)
    assert all(line[1][0] == EDGE_CONTOUR_X for line in lines)


def test_saddle_uses_deterministic_center_value_pairing():
    field = np.array([[-1.0, 1.0], [1.0, -1.0]])

    contours = contour_lines(field, level=0)

    assert len(contours) == 2
    assert all(len(line) == 2 for line in contours)
