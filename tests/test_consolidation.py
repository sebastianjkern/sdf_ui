"""Regression coverage for graph state and parameter schemas."""

import numpy as np
import pytest

from sdf_ui import Canvas, input_value, sdf
from sdf_ui.gpu.nodes.shader_expression import _primal_only
from sdf_ui.graph import GraphCase, GraphDatasetTrainer
from sdf_ui.learning import (
    Bounds,
    Element,
    GaussianPolicy,
    Layout,
    LayoutTrainer,
    Objective,
    ParameterStore,
    PolicyGradientTrainer,
)


def test_objective_terms_and_cpu_targets_are_immutable():
    target = np.ones((3, 4), dtype=np.float32)
    objective = Objective().mse("field", target, mask=target)
    target[:] = 7
    assert objective.pixel_losses[0].target[0, 0] == 1
    with pytest.raises(AttributeError):
        objective.pixel_losses.clear()
    with pytest.raises(AttributeError):
        objective.version = 0
    with pytest.raises(ValueError):
        objective.pixel_losses[0].target.flags.writeable = True
    with pytest.raises(ValueError):
        objective.pixel_losses[0].mask.flags.writeable = True
    with Canvas((4, 3)) as canvas:
        p = ParameterStore()
        radius = p.bind("radius", 1.0)
        with canvas.compile({"field": sdf.circle((2, 1), radius)}, p) as graph:
            with graph.run(objective) as execution:
                objective.set_weight(objective.pixel_losses[0].name, 2)
                with pytest.raises(RuntimeError, match="Objective changed"):
                    execution.backward()
            name = objective.pixel_losses[0].name
            version = objective.version
            objective.remove(name)
            assert objective.pixel_losses == () and objective.version == version + 1


def test_checkpoint_rejects_changed_scalar_transform_before_mutation():
    first, second = ParameterStore(), ParameterStore()
    first.scalar("x", initial=3, min=0)
    second.scalar("x", initial=12, min=10)
    with pytest.raises(ValueError, match="schema"):
        second.load_state_dict(first.state_dict())
    assert second.value("x") == pytest.approx(12)


def test_graph_checkpoint_rejects_changed_unused_parameter_schema():
    from sdf_ui.graph import GraphTrainer

    first, second = ParameterStore(), ParameterStore()
    x1 = first.scalar("x", initial=3, min=0)
    x2 = second.scalar("x", initial=3, min=0)
    first.scalar("unused", initial=3, min=0)
    second.scalar("unused", initial=12, min=10)
    with Canvas((4, 3)) as canvas:
        with (
            canvas.compile({"field": sdf.circle((2, 1), x1)}, first) as a,
            canvas.compile({"field": sdf.circle((2, 1), x2)}, second) as b,
        ):
            with (
                GraphTrainer(a, Objective().mse("field", 0)) as source,
                GraphTrainer(b, Objective().mse("field", 0)) as target,
            ):
                state = source.state_dict()
                with pytest.raises(ValueError, match="differs"):
                    target.load_state_dict(state)
                assert second.value("unused") == pytest.approx(12)


def test_zero_size_offscreen_element_is_not_valid():
    layout = Layout([Element("empty", Bounds(-5, 2, 0, 0))], (10, 10))
    assert layout.violations()["offscreen_edge_loss"] == 25
    assert not layout.is_valid()
    with pytest.raises(ValueError, match="constraint violations"):
        layout.assert_valid()


def test_graphs_share_shader_programs_but_own_scratch_buffers():
    with Canvas((4, 3)) as canvas:
        p = ParameterStore()
        radius = p.bind("radius", 1)
        field = sdf.circle((2, 1), radius)
        objective = Objective().mse("field", 0)
        with (
            canvas.compile({"field": field}, p) as first,
            canvas.compile({"field": field}, p) as second,
        ):
            with first.run(objective) as execution:
                execution.backward()
            assert canvas._pipelines
            pipeline_count = len(canvas._pipelines)
            p["radius"].zero_grad()
            with second.run(objective) as execution:
                execution.backward()
            assert len(canvas._pipelines) == pipeline_count
            buffer = second.backend.scratch_buffer("backward_partials", 4)
            p["radius"].zero_grad()
            with second.run(objective) as execution:
                execution.backward()
            assert second.backend.scratch_buffer("backward_partials", 4) is buffer
            first.close()
            p["radius"].zero_grad()
            with second.run(objective) as execution:
                execution.backward()
            assert np.isfinite(p["radius"].grad)


def test_dataset_early_stopping_restores_best_optimizer_state():
    with Canvas((4, 3)) as canvas:
        p = ParameterStore()
        radius = p.bind("radius", 1)
        with canvas.compile({"field": sdf.circle((2, 1), radius)}, p) as graph:
            case = GraphCase(graph, Objective().mse("field", 0), "training")
            with GraphDatasetTrainer([case], lr=0.05) as trainer:
                scores = iter([4.0, 1.0, 2.0, 3.0])
                values = []
                history = trainer.fit(
                    20,
                    validation=lambda: next(scores),
                    patience=2,
                    validation_interval=2,
                    callback=lambda metrics: values.append(p["radius"].item()),
                )
                assert len(history) == 6
                assert trainer.steps == 2
                assert p["radius"].item() == pytest.approx(values[1])
                assert trainer.optimizer.state_dict()["state"]["radius"]["step"] == 2


def test_dataset_extreme_weights_remain_finite():
    with Canvas((4, 3)) as canvas:
        p = ParameterStore()
        radius = p.bind("radius", 1)
        with canvas.compile({"field": sdf.circle((2, 1), radius)}, p) as graph:
            cases = [
                GraphCase(graph, Objective().mse("field", target), name, 1e308)
                for name, target in (("a", 0), ("b", 1))
            ]
            with GraphDatasetTrainer(cases) as trainer:
                before = trainer.evaluate()
                assert before["loss"] == pytest.approx(
                    (before["cases"]["a"] + before["cases"]["b"]) / 2
                )
                assert np.isfinite(trainer.step()["loss"])


def test_dataset_rejects_conflicting_inputs_for_shared_graph():
    with Canvas((4, 3)) as canvas:
        p = ParameterStore()
        radius = p.bind("radius", 1)
        with canvas.compile(
            {"field": sdf.circle((input_value("x"), 1), radius)}, p, inputs={"x": 2}
        ) as graph:
            cases = [
                GraphCase(graph, Objective().mse("field", 0), name)
                for name in ("a", "b")
            ]
            with GraphDatasetTrainer(cases) as trainer:
                state = trainer.state_dict()
                state["case_inputs"]["b"]["x"] = 3
                with pytest.raises(ValueError, match="conflicting inputs"):
                    trainer.load_state_dict(state)
                assert graph.inputs["x"] == 2


def test_primal_rewrite_respects_comments_and_nested_field_access():
    source = "// D(ignore, comment) a.d\nD f = D(g(a, D(b, x.d)), q.x.d); float v = a[1].x.d;"
    result = _primal_only(source)
    assert "// D(ignore, comment) a.d" in result
    assert "D(g(a, D(b,0.0)),0.0)" in result
    assert "float v = 0.0;" in result
    with pytest.raises(ValueError, match="Invalid dual constructor"):
        _primal_only("D(a)")


def test_layout_checkpoints_require_matching_loss_identity():
    p = ParameterStore()
    p.scalar("x", initial=1)
    missing = LayoutTrainer(p, lambda: p["x"] ** 2)
    with pytest.raises(ValueError, match="explicit loss identity"):
        missing.state_dict()
    source = LayoutTrainer(p, lambda: p["x"] ** 2, identity={"loss": "squared"})
    state = source.state_dict()
    different = LayoutTrainer(p, lambda: p["x"] ** 2, identity={"loss": "other"})
    with pytest.raises(ValueError, match="identity differs"):
        different.load_state_dict(state)


def test_named_policy_checkpoint_validates_environment_identity():
    p = ParameterStore()
    radius = p.scalar("radius", initial=1)
    context = input_value("context")
    with Canvas((4, 3)) as canvas:
        with canvas.compile(
            {"field": sdf.circle((context, 1), radius)}, p, inputs={"context": 2}
        ) as graph:
            kwargs = {
                "observations": {"context": context},
                "actions": {"radius": radius},
                "objective": Objective().mse("field", 0),
            }
            env = graph.environment(**kwargs, identity={"reward": "fit-v1"})
            trainer = PolicyGradientTrainer(GaussianPolicy(1, 1), seed=5)
            trainer.fit(env, lambda rng: {"context": 2}, steps=1, batch_size=2)
            state = trainer.state_dict()
            restored = PolicyGradientTrainer(GaussianPolicy(1, 1), seed=7)
            restored.load_state_dict(state)
            changed = graph.environment(**kwargs, identity={"reward": "different"})
            with pytest.raises(ValueError, match="environment schema differs"):
                restored.fit(changed, lambda rng: {"context": 2}, steps=0)
            with canvas.compile(
                {"field": sdf.circle((context, 1), radius)},
                p,
                inputs={"context": 2},
                size=(5, 3),
            ) as other_graph:
                other = other_graph.environment(**kwargs, identity={"reward": "fit-v1"})
                assert np.isfinite(
                    restored.evaluate(other, [{"context": 2}])["mean_reward"]
                )


def test_legacy_environment_rejects_live_execution_before_mutating():
    from sdf_ui.graph import RenderEnvironment

    p = ParameterStore()
    radius = p.bind("radius", 1)
    with Canvas((4, 3)) as canvas:
        with canvas.compile({"field": sdf.circle((2, 1), radius)}, p) as graph:
            env = RenderEnvironment(
                graph, ["radius"], objective=Objective().mse("field", 0)
            )
            with graph.run():
                with pytest.raises(RuntimeError, match="active execution"):
                    env([1], [2])
                assert p["radius"].item() == 1
