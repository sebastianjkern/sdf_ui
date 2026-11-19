from dataclasses import replace

import numpy as np
import pytest

from sdf_ui import Canvas
from sdf_ui.learning import Objective, ParameterStore
from sdf_ui.plot import (
    Geometry,
    Layer,
    Plot,
    aes,
    facet_wrap,
    geom_bar,
    geom_col,
    geom_histogram,
    geom_linerange,
    geom_point,
    geom_ribbon,
)


def test_custom_layer_inherits_mappings_and_shares_scales_across_facets():
    data = [
        {"panel": "East", "kind": "a", "day": "Mon", "temperature": 18},
        {"panel": "West", "kind": "b", "day": "Tue", "temperature": 27},
    ]
    observed = []

    def draw(plot, rows, tint):
        observed.append((plot, rows, tint))
        return None

    custom = Layer(
        Geometry(("x", "y"), draw),
        aes(y="reading"),
        data=[dict(row, reading=row["temperature"]) for row in data],
    )
    chart = (
        Plot.from_data(data, aes(x="day", color="kind")) + custom + facet_wrap("panel")
    )
    chart.build((0, 0, 720, 400))

    east, west = observed
    assert east[0].x_range == west[0].x_range == (-0.5, 1.5)
    assert east[0].y_range == west[0].y_range
    assert east[0].box[0] < west[0].box[0]
    assert east[1] == ({"x": 0, "y": 18},)
    assert west[1] == ({"x": 1, "y": 27},)
    assert east[2] != west[2]


def test_count_geometry_counts_by_category_and_group_before_fitting_scales():
    data = [
        {"day": "Mon", "kind": "a"},
        {"day": "Mon", "kind": "a"},
        {"day": "Mon", "kind": "b"},
        {"day": "Tue", "kind": "a"},
    ]
    observed = []

    def draw(plot, rows, tint):
        observed.append((plot, rows))
        return None

    bars = geom_bar(dodge=0.4)
    layer = Layer(
        Geometry(("x", "y"), draw, include_zero=True),
        dodge=bars.dodge,
        count=bars.count,
    )
    (Plot.from_data(data, aes(x="day", color="kind")) + layer).build((0, 0, 480, 320))
    first, second = observed
    assert first[0].y_range == second[0].y_range == pytest.approx((0, 2.12))
    assert first[1] == ({"x": -0.2, "y": 2}, {"x": 0.8, "y": 1})
    assert second[1] == ({"x": 0.2, "y": 1},)


def test_chart_reports_invalid_mappings_and_mixed_scale_values():
    box = (0, 0, 480, 320)
    with pytest.raises(ValueError, match="requires mappings for ymin, ymax"):
        (Plot.from_data([{"x": 1}], aes(x="x")) + geom_linerange()).build(box)
    with pytest.raises(ValueError, match="cannot mix"):
        (
            Plot.from_data([{"x": "a", "y": 1}, {"x": 0, "y": 2}], aes(x="x", y="y"))
            + geom_point()
        ).build(box)


def test_chart_mark_gradients_with_rendered_axes():
    weights = ParameterStore()
    value = weights.bind("height", 1.1)
    tint = weights.bind("tint", [0.25, 0.7, 0.6, 1])
    box = (0, 0, 220, 176)

    def layers(y, paint):
        return (
            Plot.from_data(
                [{"x": "A", "y": y, "group": "Series"}],
                aes(x="x", y="y", color="group"),
                ylim=(0, 3),
                palette=(paint,),
            )
            + geom_col(width=0.6)
        ).build(box)

    scene = layers(value, tint)
    target = layers(2, (0.5, 0.7, 0.6, 1))
    with Canvas((220, 176)) as canvas, canvas.render(scene.chrome) as chrome:
        with (
            canvas.capture(chrome.over(target.marks).to_rgb()) as image,
            canvas.compile(
                {"image": chrome.over(scene.marks).to_rgb()}, weights
            ) as graph,
        ):
            with graph.run(Objective().mse("image", image)) as execution:
                assert execution.loss > 0
                execution.backward()
            gradient = float(weights["height"].grad)
            assert np.isfinite(gradient)
            assert gradient < 0
            assert np.isfinite(weights["tint"].grad).all()
            assert np.linalg.norm(weights["tint"].grad) > 0


def test_histogram_uses_shared_bins_and_counts_the_rightmost_edge():
    observed = []

    def record(plot, rows, tint):
        observed.append((plot.x_range, rows))

    layer = geom_histogram(bins=2)
    layer = replace(layer, geometry=replace(layer.geometry, draw=record))
    rows = [
        {"panel": panel, "value": value}
        for panel, values in (("A", (0, 1, 2)), ("B", (3, 4)))
        for value in values
    ]
    plot = Plot.from_data(rows, aes(x="value")) + layer + facet_wrap("panel")
    plot.build((0, 0, 600, 250))

    assert observed[0][0] == observed[1][0] == pytest.approx((-0.24, 4.24))
    assert observed[0][1] == (
        {"xmin": 0, "xmax": 2, "ymin": 0, "ymax": 2},
        {"xmin": 2, "xmax": 4, "ymin": 0, "ymax": 1},
    )
    assert observed[1][1] == ({"xmin": 2, "xmax": 4, "ymin": 0, "ymax": 2},)


def test_ribbon_draws_a_live_band_in_the_compiled_graph():
    weights = ParameterStore()
    upper = weights.bind("upper", 1.1)
    box = (0, 0, 220, 180)

    def scene(value):
        rows = [{"x": x, "low": 0.5, "high": value} for x in range(3)]
        plot = Plot.from_data(rows, aes(x="x", ymin="low", ymax="high"), ylim=(0, 3))
        return (plot + geom_ribbon(alpha=0.7)).build(box).marks.to_rgb()

    with Canvas((220, 180)) as canvas:
        with (
            canvas.capture(scene(2)) as target,
            canvas.compile({"image": scene(upper)}, weights) as graph,
        ):
            with graph.run(Objective().mse("image", target)) as execution:
                assert execution.loss > 0
                execution.backward()
            gradient = float(weights["upper"].grad)
            assert np.isfinite(gradient)
            assert gradient < 0
