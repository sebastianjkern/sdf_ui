"""Public graph facade, diagnostics and resource ownership."""

import importlib.util

import numpy as np
import pytest

from sdf_ui import Canvas, sdf
from sdf_ui.backends.opengl import GPUImage
from sdf_ui.graph import (
    GraphPlan,
    GraphTrainer,
    RenderGraph,
    inspect_graph,
    lower_graph,
)
from sdf_ui.learning import Objective, ParameterStore
from sdf_ui.plugins import (
    Plugin,
    PluginFamily,
    Port,
    ShaderArgument,
    ShaderDifferentiation,
    ShaderFile,
    TextureKind,
    registry,
)


@pytest.mark.parametrize(
    "module",
    [
        "sdf_ui.learning.graph",
        "sdf_ui.learning.graph_lowering",
        "sdf_ui.learning.plan_execution",
        "sdf_ui.learning.gpu_backend",
        "sdf_ui.learning.gpu_glsl",
        "sdf_ui.learning.gpu_primitives",
        "sdf_ui.learning.gpu_shading",
        "sdf_ui.learning.gpu_effects",
        "sdf_ui.core.shape_programs",
    ],
)
def test_removed_compatibility_modules_are_not_importable(module):
    assert importlib.util.find_spec(module) is None


def test_graph_api_is_exposed_only_under_current_namespaces():
    import sdf_ui.core.context as context
    import sdf_ui.learning as learning

    assert not hasattr(learning, "GraphTrainer")
    assert not hasattr(learning, "GPUImage")
    assert not hasattr(learning, "lower_graph")
    assert not hasattr(Canvas, "compile_graph")
    assert not hasattr(Canvas, "local_size")
    assert not hasattr(context, "init_sdf_ui")
    assert not hasattr(context, "decrease_tex_registry")


def test_parameter_bind_is_reusable_and_named():
    params = ParameterStore()
    radius = params.bind("radius", 2.0)
    assert params.bind("radius") == radius
    with pytest.raises(ValueError, match="already exists"):
        params.bind("radius", 3)
    with pytest.raises(KeyError, match="initial value"):
        params.bind("missing")
    scene = sdf.circle((4, 3), radius.softplus())
    assert not inspect_graph({"field": scene}, params={})["supported"]
    assert inspect_graph({"field": scene}, params=params)["required_parameters"] == [
        "radius"
    ]
    with Canvas((8, 6)) as canvas, canvas.compile({"field": scene}, params) as graph:
        with graph.run(Objective().mse("field", 0)) as execution:
            execution.backward()
        assert np.isfinite(params["radius"].grad)
        params["radius"].assign(3)
        with graph.run() as execution:
            assert np.all(np.isfinite(execution.read("field")))


def test_unified_render_backends_and_owned_training_images():
    params = ParameterStore()
    radius = params.bind("r", 2.0)
    scene = sdf.circle((4, 3), radius)
    plan = lower_graph({"field": scene})
    assert isinstance(plan, GraphPlan)
    assert Plugin and Port and ShaderFile
    with Canvas((8, 6)) as canvas, canvas.compile(plan, params) as graph:
        assert isinstance(graph, RenderGraph)
        reference = graph.render(backend="reference")["field"].numpy()
        display = graph.render(backend="display")["field"]
        np.testing.assert_allclose(
            np.frombuffer(display.tex.read(), np.float32).reshape(6, 8),
            reference,
            atol=1e-5,
        )
        with graph.render(backend="training")["field"] as image:
            np.testing.assert_allclose(image.read(), reference, atol=1e-5)
        with pytest.raises(ValueError, match="backend"):
            graph.render(backend="fallback")


def test_inspection_exposes_zero_gradient_and_topology_controls():
    params = ParameterStore()
    radius = params.bind("radius", 2)
    spacing = params.bind("spacing", 3)
    field = sdf.circle((4, 3), radius).repeat(spacing)
    with Canvas((8, 6)) as canvas, canvas.compile({"field": field}, params) as graph:
        report = graph.explain()
        assert report["supported"]
        assert report["parameters"]["radius"]["continuous_uniforms"] == [
            {"shader": "circle", "uniform": "radius"}
        ]
        assert report["parameters"]["spacing"]["zero_gradient_uniforms"] == [
            {"shader": "repeat", "uniform": "repeat"}
        ]
        assert any(op["shader"] == "repeat" for op in report["operations"])
        unsupported = canvas.inspect_graph(
            {"field": sdf.polygon(((0, 0), (4, 0), (1, 3)))}
        )
        assert (
            not unsupported["supported"]
            and "custom render function" in unsupported["error"]
        )
        image = field.fill("#fff").to_rgb().blur(n=2)
        report = inspect_graph({"image": image})
        assert {item["operation"] for item in report["topology_controls"]} >= {"blur"}


def test_inspection_reports_missing_gpu_derivative(tmp_path, monkeypatch):
    path = tmp_path / "custom.glsl"
    path.write_text("#version 430\nlayout(local_size_x=1) in;\nvoid main(){}\n")
    name = "inspection_missing_derivative"
    contract = ShaderDifferentiation(
        name, str(path), (), (), (ShaderArgument("destTex", (), "r32f"),)
    )
    descriptor = ShaderFile(name, str(path), derivative_contract=contract)
    plugin = Plugin(name, PluginFamily.PRIMITIVE, TextureKind.SDF, shader=descriptor)
    registry.ensure_loaded()
    monkeypatch.setitem(registry._plugins, name, plugin)
    report = inspect_graph({"field": registry.build(name)})
    assert not report["supported"]
    assert "backward" in report["error"]


def test_backends_share_snapshots_of_materialized_sources():
    from sdf_ui.core.color import ColorTexture

    with Canvas((4, 3)) as canvas:
        texture = canvas.rgba8()
        first = np.full((3, 4, 4), 50, np.uint8)
        second = np.full((3, 4, 4), 200, np.uint8)
        texture.write(first.tobytes())
        source = ColorTexture(tex=texture, context=canvas, mode="RGB")
        with canvas.compile({"image": source}) as graph:
            texture.write(second.tobytes())
            for expected in (first, second):
                display = graph.render(backend="display")["image"]
                np.testing.assert_array_equal(
                    np.frombuffer(display.tex.read(), np.uint8).reshape(3, 4, 4),
                    expected,
                )
                reference = graph.render(backend="reference")["image"].numpy()
                np.testing.assert_allclose(reference, expected / 255.0, atol=1e-6)
                with graph.render(backend="training")["image"] as result:
                    np.testing.assert_allclose(
                        result.read(), expected / 255.0, atol=1e-6
                    )
                graph.refresh_inputs()


def test_capture_scene_as_target_without_pixel_readback(monkeypatch):
    def no_readback(*args, **kwargs):
        raise AssertionError("capture or training read pixels back to the CPU")

    monkeypatch.setattr(GPUImage, "read", no_readback)
    parameters = ParameterStore()
    radius = parameters.bind("radius", 1.0)
    with Canvas((8, 6)) as canvas:
        with canvas.capture(sdf.circle((4, 3), 3.0)) as target:
            assert not target.closed
            objective = Objective().mse("field", target)
            with canvas.compile(
                {"field": sdf.circle((4, 3), radius)}, parameters
            ) as graph:
                with GraphTrainer(graph, objective, lr=0.2) as trainer:
                    metrics = trainer.fit(70)
                    assert metrics[-1]["loss"] < metrics[0]["loss"] * 0.01
                    assert graph.backend.stats["image_readbacks"] == 0
        assert target.closed


def test_capture_color_scene_and_release_on_exception():
    from sdf_ui import color

    with Canvas((4, 3)) as canvas:
        with pytest.raises(RuntimeError, match="test failure"):
            with canvas.capture(color.clear("#abcdef").to_rgb()) as target:
                assert target.shape == (3, 4, 4)
                raise RuntimeError("test failure")
        assert target.closed
