import math

import numpy as np
import pytest

from examples.lagrange_intuition_short import PLOT, _level_curve
from sdf_ui import Canvas, Plot, sdf


@pytest.mark.parametrize("scale", [1, 2])
def test_polyline_matches_segment_projection_in_canvas_pixels(scale):
    paths = [
        [(-2, 5), (18, 5), (18, 17), (10, 23), (5, 19), (5, 19)],
        [(20, 18), (26, 28), (35, 6), (28, 5)],
    ]
    radius = 1.25
    node = sdf.polyline(
        [*np.asarray(paths[0], dtype=np.float32), None, *paths[1]],
        radius=np.float32(radius),
    )
    with Canvas((36, 30), device_scale=scale) as canvas, canvas.session() as renderer:
        width, height = canvas.size
        field = np.frombuffer(renderer.render(node).tex.read(), dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack((xx, yy), axis=-1)
    distances = []
    for path in paths:
        positions = np.array(path) * scale
        for start, end in zip(positions, positions[1:]):
            edge = end - start
            t = np.clip((pixels - start) @ edge / max(edge @ edge, 1), 0, 1)
            distances.append(
                np.linalg.norm(pixels - start - t[..., None] * edge, axis=-1)
            )
    expected = np.min(distances, axis=0) - radius * scale
    np.testing.assert_allclose(field.reshape(expected.shape), expected, atol=1e-5)


def test_curve_clips_centerline_to_nonuniform_viewport():
    plot = Plot((12, 8, 80, 45), (-2, 2), (-1, 3))
    points = dict(plot.curve(lambda x: 2 * x, domain=(-3, 3), samples=3).params)[
        "points"
    ]
    assert points[0] == pytest.approx((42, 8))
    assert points[-1] == pytest.approx((82, 53))
    assert all(12 <= x <= 92 and 8 <= y <= 53 for x, y in points)


@pytest.mark.parametrize("explicit_break", [False, True])
def test_curve_does_not_draw_across_a_jump(explicit_break):
    plot = Plot((0, 0, 120, 100), (-1, 1), (-2, 2))
    node = plot.curve(
        lambda x: 1 if x >= 0.2 else -1, breaks=(0.2,) if explicit_break else ()
    )
    points = dict(node.params)["points"]
    assert points.count(None) == 1
    split = points.index(None)
    assert all(y == 25 for x, y in points[:split])
    assert all(y == 75 for x, y in points[split + 1 :])


def test_asymptotes_and_undefined_samples_leave_gaps():
    plot = Plot((0, 0, 120, 100), (-2, 2), (-2, 2))
    # The pole falls between seed samples.
    node = plot.curve(lambda x: 1 / (x - 0.17), samples=9)
    points = dict(node.params)["points"]
    assert points.count(None) == 1
    split = points.index(None)
    assert points[split - 1][1] == 0
    assert points[split + 1][1] == 100
    empty = plot.curve(lambda x: math.sqrt(x - 3))
    assert dict(empty.params)["points"] == ()
    with Canvas((8, 8)) as canvas, canvas.session() as renderer:
        field = np.frombuffer(renderer.render(empty).tex.read(), dtype=np.float32)
    assert np.all(np.isfinite(field) & (field > 100))


def test_parametric_curve_can_loop():
    plot = Plot((0, 0, 100, 100), (-1, 1), (-1, 1))
    node = plot.parametric(
        lambda t: (math.cos(t), math.sin(t)), (0, math.tau), samples=4
    )
    points = dict(node.params)["points"]
    assert None not in points
    assert points[0] == pytest.approx(points[-1])
    assert (0, 50) == pytest.approx(points[len(points) // 2])
    assert len(points) > 5


@pytest.mark.parametrize("level", [0.08, 0.16, 0.24, 0.32, 0.40, 0.46, 0.5])
def test_lagrange_contour_is_monotone_and_subpixel_accurate(level):
    curve = _level_curve(level, "#ffffff").inputs[0]
    points = np.asarray(dict(curve.params)["points"])
    assert np.all(np.diff(points[:, 0]) > 0)
    assert np.all(np.diff(points[:, 1]) < 0)
    x = np.linspace(points[0, 0], points[-1, 0], 4096)
    left, bottom, width, height = PLOT.box
    graph_x = (x - left) / width * PLOT.x_range[1]
    true_y = bottom + level / graph_x / PLOT.y_range[1] * height
    sampled_y = np.interp(x, points[:, 0], points[:, 1])
    assert np.max(np.abs(sampled_y - true_y)) <= 0.25
