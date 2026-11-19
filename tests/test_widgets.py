import numpy as np
import pytest

from sdf_ui import Canvas, Plot, color, param, sdf
from sdf_ui.graph import GraphTrainer
from sdf_ui.learning import (
    Bounds,
    Element,
    Layout,
    Objective,
    Parameter,
    ParameterStore,
)
from sdf_ui.widgets import Panel, progress_bar


def _rgba(texture):
    width, height = texture.tex.size
    return np.frombuffer(texture.tex.read(), dtype=np.uint8).reshape(height, width, 4)


def test_panel_uses_full_dimensions_and_can_clip_its_content():
    panel = Panel((10, 7, 24, 22), padding=(4, 3), radius=7)
    assert panel.content_box == (14, 10, 16, 16)
    assert Panel(panel.box, padding=(20, 15)).content_box == (22, 18, 0, 0)
    background = "#00000000"
    inside = panel.layer(background, color.clear("#ffffff"), clip=True)
    with Canvas((48, 36)) as canvas, canvas.render(inside) as image:
        pixels = _rgba(image)
        assert pixels[18, 22, 3] == 255
        assert pixels[7, 10, 3] == 0
        assert pixels[18, 36, 3] == 0


def test_progress_bar_clamps_a_live_value_and_keeps_zero_empty():
    bar = progress_bar(
        (4, 4, 28, 11), param("value"), track_color="#0000ff", fill_color="#ff0000"
    )
    with Canvas((36, 18)) as canvas:

        def render(value):
            with canvas.render(bar, params={"value": value}) as image:
                return _rgba(image)

        empty = render(0)
        np.testing.assert_array_equal(render(-1), empty)
        assert empty[9, 10, 2] > 200
        small = render(0.15)
        assert np.all(small[..., 3][empty[..., 3] == 0] == 0)
        half = render(0.5)
        assert half[9, 10, 0] > 200
        assert half[9, 24, 2] > 200
        full = render(1)
        np.testing.assert_array_equal(render(2), full)
        assert full[9, 24, 0] > 200


def test_arrow_handles_short_and_coincident_points_in_a_compiled_graph():
    endpoint = Parameter("x", 17.0)
    arrow = sdf.arrow((9, 8), (endpoint, 8), width=4, head_length=20)
    with Canvas((32, 22)) as canvas, canvas.compile({"arrow": arrow}) as graph:
        with graph.run() as evaluation:
            extended = evaluation.read("arrow")
        assert extended[8, 15] < 0
        endpoint.assign(9.0)
        with graph.run() as evaluation:
            collapsed = evaluation.read("arrow")
    y, x = np.indices(collapsed.shape)
    np.testing.assert_allclose(collapsed, np.hypot(x - 9, y - 8) - 2, atol=2e-6)


def test_annotation_anchors_a_text_baseline_and_draws_a_leader():
    plot = Plot((4, 4, 32, 20), (0, 8), (0, 5))
    label = plot.annotate((2, 2), "A", offset=(12, 0), leader_width=2, size=14)
    with Canvas((44, 34)) as canvas, canvas.render(label) as image:
        pixels = _rgba(image)
    assert pixels[12, 18, 3] > 150
    assert pixels[18:24, 25:34, 3].max() > 200
    assert pixels[12, 7, 3] == 0


def test_widget_geometry_can_share_layout_losses_and_native_pixel_gradients():
    params = ParameterStore(
        [
            Parameter("width", 15.0),
            Parameter("progress", 0.35),
            Parameter("tip", 18.0),
            Parameter("offset", 6.0),
            Parameter("leader_width", 2.0),
        ]
    )
    size = (40, 30)
    plot = Plot((4, 4, 32, 20), (0, 8), (0, 5))

    def scene(label, width, progress, tip, offset, leader_width):
        panel = Panel((3, 5, width, 14), padding=(2, 3), radius=2)
        return panel, {
            "panel": panel.shape(),
            "progress": progress_bar((4, 22, 30, 5), progress),
            "arrow": sdf.arrow((7, 12), (tip, 15), head_length=7),
            "annotation": plot.annotate(
                (2, 2),
                label,
                label_origin=(8, 8),
                offset=(offset, 6),
                leader_width=leader_width,
            ),
        }

    with (
        Canvas(size) as canvas,
        canvas.render(sdf.text("A", size=12, ox=8, oy=8)) as label,
    ):
        panel, live = scene(
            label,
            param("width"),
            param("progress"),
            param("tip"),
            param("offset"),
            param("leader_width"),
        )
        _, target = scene(label, 19, 0.68, 22, 9, 3.5)
        layout = Layout(
            [
                Element(
                    "card",
                    Bounds(*panel.box),
                    minimum_size=(18, 10),
                    content_bounds=Bounds(*panel.content_box),
                )
            ],
            size,
        )
        with (
            canvas.compile(live, params=params) as graph,
            canvas.compile(target) as reference,
        ):
            with reference.run() as expected:
                objective = Objective()
                for name in live:
                    objective.mse(name, expected.read(name))
            objective.geometry(lambda: layout.minimum_size_loss(params), weight=0.1)
            with graph.run(objective) as run:
                gradients = run.backward()
                assert run.stats["image_readbacks"] == 0
            for name in params:
                assert float(gradients[name]) != pytest.approx(0, abs=1e-6)
                assert np.isfinite(gradients[name])
            with GraphTrainer(graph, objective, lr=0.08) as trainer:
                history = trainer.fit(4)
            assert history[-1]["loss"] < history[0]["loss"]
