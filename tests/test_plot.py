
import math

import numpy as np
import pytest

from examples.lagrange_intuition_short import PLOT, _level_curve
from sdf_ui import Canvas, Plot, sdf
from sdf_ui.graph import GraphTrainer
from sdf_ui.learning import Objective, ParameterStore

_POINTS_TEXT = 'points'

def test_plot_maps_both_directions_with_independent_scales():
    plot = Plot((12, 8, 80, 45), (-2, 2), (-1, 3))
    assert plot.point(-2, -1) == (12, 8)
    assert plot.point(2, 3) == (92, 53)
    assert plot.data_point(*plot.point(3, -0.5)) == pytest.approx((3, -0.5))


def test_ribbon_has_no_ink_seam_at_a_join_and_preserves_gaps():
    plot = Plot((4, 4, 24, 20), (0, 12), (0, 10))
    samples = [(1, 3, 7), (2, 3, 7), (4, 3, 7), None, (10, 3, 7), (8, 3, 7)]
    with Canvas((32, 28)) as canvas, canvas.render(plot.ribbon(samples)) as output:
        field = np.frombuffer(output.tex.read(), dtype=np.float32).reshape(28, 32)
    assert field[14, 8] <= -0.5  # First run's internal join.
    assert field[14, 16] > 0  # Gap between runs.
    assert field[14, 22] < 0  # Second run travels right to left.


def test_axes_format_auto_ticks_and_validate_crossings():
    plot = Plot((30, 30, 100, 80), (0, 1), (-2, 4))
    formatted = []

    def format_y(value):
        formatted.append(value)
        return f"{value:g} V"

    plot.axes(x_ticks=(), y_ticks=None, y_format=format_y, x_axis_at=2)
    assert formatted == [-2, -1, 0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="y_axis_at"):
        plot.axes(y_axis_at=2)


def test_data_series_preserve_gaps_and_signed_bar_geometry():
    plot = Plot((4, 8, 32, 24), (0, 4), (-2, 2))
    path = plot.line([(0, 0), (1, 1), None, (2, -1), (3, 0)], radius=1)
    first, second = path.inputs
    assert dict(first.params)["b"] == plot.point(1, 1)
    assert dict(second.params)["a"] == plot.point(2, -1)
    bars = plot.bars([(1, -1), (2, 1)], width=0.5)
    for node, x, y in zip(bars.inputs, (1, 2), (-0.5, 0.5)):
        params = dict(node.params)
        assert params["center"] == plot.point(x, y)
        assert params["size"] == (2, 3)


def test_series_train_through_shared_data_coordinate():
    plot = Plot((3, 3, 26, 22), (0, 2), (-1, 1))
    weights = ParameterStore()
    value = weights.bind("value", 0.2)

    def roots(y):
        points = [(0.5, -7 / 20), (1.5, y)]
        line = plot.clip(plot.line(points, radius=1))
        return {
            "bars": plot.clip(plot.bars(points, width=0.5)),
            "line": line,
            "markers": plot.scatter(points, radius=2).ink("#66e0c5").to_rgb(),
        }

    with Canvas((32, 28)) as canvas:
        with canvas.compile(roots(0.55)) as target:
            with (
                target.snapshot("bars") as bars,
                target.snapshot("line") as line,
                target.snapshot("markers") as markers,
                canvas.compile(roots(value), weights) as graph,
            ):
                objective = (
                    Objective()
                    .mse("bars", bars)
                    .mse("line", line)
                    .mse("markers", markers)
                )
                with graph.run(objective) as execution:
                    initial = execution.loss
                    execution.backward()
                assert np.isfinite(weights["value"].grad).all()
                assert abs(float(weights["value"].grad)) > 0
                with GraphTrainer(graph, objective, lr=0.04) as trainer:
                    trainer.fit(6)
                with graph.run(objective) as execution:
                    assert execution.loss < initial


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
    expected = _polyline_distance_field(paths, width, height, radius, scale)
    np.testing.assert_allclose(field.reshape(expected.shape), expected, atol=1e-5)


def _polyline_distance_field(paths, width, height, radius, scale):
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack((xx, yy), axis=-1)
    distances = [
        _segment_distances(pixels, path, scale)
        for path in paths
    ]
    return np.min([item for group in distances for item in group], axis=0) - radius * scale


def _segment_distances(pixels, path, scale):
    positions = np.array(path) * scale
    distances = []
    for start, end in zip(positions, positions[1:]):
        edge = end - start
        t = np.clip((pixels - start) @ edge / max(edge @ edge, 1), 0, 1)
        distances.append(np.linalg.norm(pixels - start - t[..., None] * edge, axis=-1))
    return distances


def test_curve_clips_centerline_to_nonuniform_viewport():
    plot = Plot((12, 8, 80, 45), (-2, 2), (-1, 3))
    points = dict(plot.curve(lambda x: 2 * x, domain=(-3, 3), samples=3).params)[
        _POINTS_TEXT
    ]
    assert points[0] == pytest.approx((42, 8))
    assert points[-1] == pytest.approx((82, 53))
    assert all(12 <= x <= 92 and 8 <= y <= 53 for x, y in points)


@pytest.mark.parametrize("explicit_break", [False, True])
def test_curve_does_not_draw_across_a_jump(explicit_break):
    plot = Plot((0, 0, 120, 100), (-1, 1), (-2, 2))
    node = plot.curve(
        lambda x: 1 if x >= 1 / 5 else -1, breaks=(0.2,) if explicit_break else ()
    )
    points = dict(node.params)[_POINTS_TEXT]
    assert points.count(None) == 1
    split = points.index(None)
    assert all(y == 25 for x, y in points[:split])
    assert all(y == 75 for x, y in points[split + 1 :])


def test_asymptotes_and_undefined_samples_leave_gaps():
    plot = Plot((0, 0, 120, 100), (-2, 2), (-2, 2))
    # The pole falls between seed samples.
    node = plot.curve(lambda x: 1 / (x - 17 / 100), samples=9)
    points = dict(node.params)[_POINTS_TEXT]
    assert points.count(None) == 1
    split = points.index(None)
    assert points[split - 1][1] == 0
    assert points[split + 1][1] == 100
    empty = plot.curve(lambda x: math.sqrt(x - 3))
    assert dict(empty.params)[_POINTS_TEXT] == ()
    with Canvas((8, 8)) as canvas, canvas.session() as renderer:
        field = np.frombuffer(renderer.render(empty).tex.read(), dtype=np.float32)
    assert np.all(np.isfinite(field) & (field > 100))


def test_parametric_curve_can_loop():
    plot = Plot((0, 0, 100, 100), (-1, 1), (-1, 1))
    node = plot.parametric(
        lambda t: (math.cos(t), math.sin(t)), (0, math.tau), samples=4
    )
    points = dict(node.params)[_POINTS_TEXT]
    assert None not in points
    assert points[0] == pytest.approx(points[-1])
    assert (0, 50) == pytest.approx(points[len(points) // 2])
    assert len(points) > 5


@pytest.mark.parametrize("level", [0.08, 0.16, 0.24, 0.32, 0.40, 0.46, 0.5])
def test_lagrange_contour_is_monotone_and_subpixel_accurate(level):
    curve = _level_curve(level, "#ffffff").inputs[0]
    points = np.asarray(dict(curve.params)[_POINTS_TEXT])
    assert np.all(np.diff(points[:, 0]) > 0)
    assert np.all(np.diff(points[:, 1]) < 0)
    x = np.linspace(points[0, 0], points[-1, 0], 4096)
    left, bottom, width, height = PLOT.box
    graph_x = (x - left) / width * PLOT.x_range[1]
    true_y = bottom + level / graph_x / PLOT.y_range[1] * height
    sampled_y = np.interp(x, points[:, 0], points[:, 1])
    assert np.max(np.abs(sampled_y - true_y)) <= 1 / 4
