"""Public learning workflows, ownership, and numerical behavior."""



import numpy as np
import pytest

from sdf_ui import Canvas, input_value, sdf
from sdf_ui.graph import GraphCase, GraphDatasetTrainer, GraphTrainer
from sdf_ui.learning import (
    Bounds,
    CategoricalPolicy,
    Element,
    GaussianPolicy,
    Layout,
    LayoutTrainer,
    Objective,
    Parameter,
    ParameterStore,
    PolicyGradientTrainer,
)

_CONTEXT_TEXT = 'context'

_FIELD_TEXT = 'field'

_OFFSET_TEXT = 'offset'

_RADIUS_TEXT = 'radius'

@pytest.mark.parametrize(
    "bounds,initial,next_value",
    [
        ({}, 3.0, -1.0),
        ({"min": 0}, 3.0, 0.2),
        ({"max": 5}, 3.0, 4.8),
        ({"min": 0, "max": 5}, 3.0, 4.8),
    ],
)
def test_scalar_uses_actual_rendered_units(bounds, initial, next_value):
    weights = ParameterStore()
    radius = weights.scalar(_RADIUS_TEXT, initial=initial, **bounds)
    assert weights.value(_RADIUS_TEXT) == pytest.approx(initial)
    with Canvas((3, 3)) as canvas:
        with canvas.compile({_FIELD_TEXT: sdf.circle((1, 1), radius)}, weights) as graph:
            with graph.run() as execution:
                assert execution.read(_FIELD_TEXT)[1, 1] == pytest.approx(
                    -initial, abs=1e-6
                )
            weights.assign_value(_RADIUS_TEXT, next_value)
            with graph.run() as execution:
                assert execution.read(_FIELD_TEXT)[1, 1] == pytest.approx(
                    -next_value, abs=1e-6
                )
    if bounds:
        with pytest.raises(ValueError, match="strictly inside"):
            weights.assign_value(_RADIUS_TEXT, bounds.get("min", bounds.get("max")))


def _training_scene(canvas, field, weights):
    return canvas.training(
        {_FIELD_TEXT: field, "image": field.fill("#ffffff", "#000000").to_rgb()},
        weights,
    )


def _train_and_save_session(canvas, weights, field, tmp_path):
    with _training_scene(canvas, field, weights) as session:
        session.match(sdf.circle((4, 4), 3), output=_FIELD_TEXT)
        initial = session.evaluate()["loss"]
        session.fit(40, lr=0.1)
        assert session.evaluate()["loss"] < initial * 1 / 20
        _assert_session_report(session, weights)
        session.save(tmp_path / "result.png", output="image")
        session.save_checkpoint(tmp_path / "state.json")
        saved = weights.value(_RADIUS_TEXT)
        weights.assign_value(_RADIUS_TEXT, 1)
        session.load_checkpoint(tmp_path / "state.json")
        assert weights.value(_RADIUS_TEXT) == pytest.approx(saved)
    assert session.graph.closed
    return saved


def _assert_session_report(session, weights):
    report = session.report()
    assert report["parameters"][_RADIUS_TEXT] == pytest.approx(
        weights.value(_RADIUS_TEXT)
    )
    assert report["disconnected_parameters"] == []


def _resume_training_session(canvas, weights, field, tmp_path, saved):
    with _training_scene(canvas, field, weights) as resumed:
        resumed.match(sdf.circle((4, 4), 3), output=_FIELD_TEXT)
        resumed.load_checkpoint(tmp_path / "state.json")
        assert weights.value(_RADIUS_TEXT) == pytest.approx(saved)
        assert resumed.fit(1, lr=0.1)[0]["step"] == 41


def test_session_owns_targets_and_trains(tmp_path):
    weights = ParameterStore()
    radius = weights.scalar(_RADIUS_TEXT, initial=1, min=0)
    field = sdf.circle((4, 4), radius)
    with Canvas((8, 8)) as canvas:
        saved = _train_and_save_session(canvas, weights, field, tmp_path)
        assert (tmp_path / "result.png").exists()
        _resume_training_session(canvas, weights, field, tmp_path, saved)


def test_mask_normalization_uses_gpu_reduction_and_correct_gradient():
    with Canvas((4, 3)) as canvas:
        weights = ParameterStore()
        radius = weights.scalar(_RADIUS_TEXT, initial=1)
        with canvas.compile({_FIELD_TEXT: sdf.circle((2, 1), radius)}, weights) as graph:
            mask = np.zeros((3, 4), dtype=np.float32)
            mask[0, 0], mask[0, 1] = 0.5, 1
            with graph.backend.upload(mask) as gpu_mask:
                objective = Objective().mse(
                    _FIELD_TEXT, 0, mask=gpu_mask, normalization="mask"
                )
                with graph.run(objective) as execution:
                    pixels = execution.read(_FIELD_TEXT)
                    execution.backward()
                    assert execution.loss == pytest.approx(
                        (pixels**2 * mask).sum() / mask.sum(), abs=1e-6
                    )
                    assert weights[_RADIUS_TEXT].grad == pytest.approx(
                        (-2 * pixels * mask).sum() / mask.sum(), abs=1e-6
                    )
                zero = Objective().mse(_FIELD_TEXT, 0, mask=0, normalization="mask")
                with pytest.raises(ValueError, match="positive sum"):
                    graph.run(zero)


def test_dynamic_policy_observations_and_checkpoint(tmp_path):
    policy = GaussianPolicy(1, 1, hidden_dim=4, seed=5)
    observation = input_value(_CONTEXT_TEXT)
    radius = policy.mean_expressions([observation])[0].softplus()
    with Canvas((4, 3)) as canvas:
        with canvas.compile(
            {_FIELD_TEXT: sdf.circle((2, 1), radius)}, inputs={_CONTEXT_TEXT: 0.1}
        ) as graph:
            objective = Objective().mse(_FIELD_TEXT, 0)
            first = _check_active_input_guard(graph, objective)
            graph.set_inputs(context=0.8)
            _check_updated_policy_render(graph, objective, policy, first)
            _check_policy_checkpoint(graph, objective, tmp_path)


def _check_active_input_guard(graph, objective):
    with graph.run(objective) as execution:
        first = execution.loss
        with pytest.raises(RuntimeError, match="active execution"):
            graph.set_inputs(context=0.8)
    return first


def _check_updated_policy_render(graph, objective, policy, first):
    with graph.run(objective) as execution:
        assert execution.loss != pytest.approx(first, abs=1e-7)
        expected = execution.read(_FIELD_TEXT)
        execution.backward()
        assert all(parameter.grad is not None for parameter in policy.parameters())
    reference = graph.render(backend="reference")[_FIELD_TEXT].numpy()
    display = graph.render(backend="display")[_FIELD_TEXT]
    np.testing.assert_allclose(reference, expected, atol=1e-5)
    np.testing.assert_allclose(
        np.frombuffer(display.tex.read(), np.float32).reshape(3, 4),
        expected,
        atol=1e-5,
    )
    return expected


def _check_policy_checkpoint(graph, objective, tmp_path):
    with GraphTrainer(graph, objective) as trainer:
        trainer.save(tmp_path / "policy.json")
        graph.set_inputs(context=0.3)
        trainer.load(tmp_path / "policy.json")
        assert graph.inputs[_CONTEXT_TEXT] == 4 / 5


def test_named_environment_policy_fit_and_bounded_actions():
    weights = ParameterStore()
    radius = weights.scalar(_RADIUS_TEXT, initial=1, min=0, max=5)
    context = input_value(_CONTEXT_TEXT)
    with Canvas((4, 3)) as canvas:
        with canvas.compile(
            {_FIELD_TEXT: sdf.circle((context, 1), radius)}, weights, inputs={_CONTEXT_TEXT: 2}
        ) as graph:
            seen = []

            def reward(obs, action, execution):
                seen.append((obs.context, action.radius, type(action.radius)))
                return -execution.loss

            env = graph.environment(
                observations={_CONTEXT_TEXT: context},
                actions={_RADIUS_TEXT: radius},
                objective=Objective().mse(_FIELD_TEXT, 0),
                reward=reward,
                action_transform=lambda action: {
                    _RADIUS_TEXT: 1 / 10 + 24 / 5 / (1 + np.exp(-action[0]))
                },
            )
            assert np.isfinite(env({_CONTEXT_TEXT: 1.5}, {_RADIUS_TEXT: 2}))
            assert weights.value(_RADIUS_TEXT) == pytest.approx(2)
            assert graph.inputs[_CONTEXT_TEXT] == 3 / 2
            assert seen[-1] == (1.5, 2, float)
            trainer = PolicyGradientTrainer(GaussianPolicy(1, 1, seed=4), seed=9)
            history = trainer.fit(
                env, lambda rng: {_CONTEXT_TEXT: 1.5}, steps=2, batch_size=4
            )
            assert len(history) == 2
            assert np.isfinite(trainer.evaluate(env, [{_CONTEXT_TEXT: 1.5}])["mean_reward"])
            action = env.act(trainer.policy, {_CONTEXT_TEXT: 1.5})
            assert isinstance(action.radius, float)
            assert 0 < action.radius < 5


def test_layout_validation_and_trainer_checkpoint(tmp_path):
    x = Parameter("x", -2)
    layout = Layout([Element("item", Bounds(x, 1, 2, 2))], (5, 5))
    assert not layout.is_valid()
    with pytest.raises(ValueError, match="constraint violations"):
        layout.assert_valid()
    assert layout.prevent_overlap().item() == 0
    trainer = LayoutTrainer(
        [x], lambda: layout.offscreen_loss(), identity="offscreen-v1"
    )
    initial = trainer.evaluate()["loss"]
    trainer.fit(2)
    assert trainer.evaluate()["loss"] < initial
    trainer.save(tmp_path / "layout.json")
    saved = x.item()
    trainer.fit(2)
    trainer.load(tmp_path / "layout.json")
    assert x.item() == saved
    assert trainer.steps == 2


def test_dataset_checkpoint_restores_each_graph_input():
    weights = ParameterStore()
    radius = weights.scalar(_RADIUS_TEXT, initial=1)
    offset = input_value(_OFFSET_TEXT)
    with Canvas((4, 3)) as canvas:
        with (
            canvas.compile(
                {_FIELD_TEXT: sdf.circle((offset, 1), radius)},
                weights,
                inputs={_OFFSET_TEXT: 1},
            ) as first,
            canvas.compile(
                {_FIELD_TEXT: sdf.circle((offset, 1), radius)},
                weights,
                inputs={_OFFSET_TEXT: 3},
            ) as second,
        ):
            cases = [
                GraphCase(first, Objective().mse(_FIELD_TEXT, 0), "first"),
                GraphCase(second, Objective().mse(_FIELD_TEXT, 1), "second"),
            ]
            with GraphDatasetTrainer(cases) as trainer:
                trainer.fit(1)
                state = trainer.state_dict()
                first.set_inputs(offset=0)
                second.set_inputs(offset=0)
                trainer.load_state_dict(state)
                assert first.inputs[_OFFSET_TEXT] == 1
                assert second.inputs[_OFFSET_TEXT] == 3


def test_named_categorical_actions_use_a_python_integer():
    weights = ParameterStore()
    radius = weights.scalar(_RADIUS_TEXT, initial=1)
    context = input_value(_CONTEXT_TEXT)
    seen = []
    with Canvas((4, 3)) as canvas:
        with canvas.compile(
            {_FIELD_TEXT: sdf.circle((context, 1), radius)}, weights, inputs={_CONTEXT_TEXT: 2}
        ) as graph:

            def choose(action):
                seen.append(type(action))
                return {_RADIUS_TEXT: (1, 2)[action]}

            environment = graph.environment(
                observations={_CONTEXT_TEXT: context},
                actions={_RADIUS_TEXT: radius},
                objective=Objective().mse(_FIELD_TEXT, 0),
                action_transform=choose,
            )
            trainer = PolicyGradientTrainer(CategoricalPolicy(1, 2), seed=3)
            trainer.fit(
                environment, lambda random: {_CONTEXT_TEXT: 2}, steps=1, batch_size=3
            )
            assert seen and all(kind is int for kind in seen)
            assert environment.act(trainer.policy, {_CONTEXT_TEXT: 2}).radius in (1, 2)


def test_layout_can_follow_dynamic_graph_inputs():
    weights = ParameterStore()
    x = weights.scalar("x", initial=1)
    offset = input_value(_OFFSET_TEXT)
    with Canvas((4, 3)) as canvas:
        with canvas.compile(
            {_FIELD_TEXT: sdf.circle((x + offset, 1), 1)}, weights, inputs={_OFFSET_TEXT: 0}
        ) as graph:
            layout = Layout(
                [Element("item", Bounds(x + offset, 0, 2, 2))],
                (4, 3),
                inputs=graph.inputs,
            )
            assert layout.is_valid(weights)
            graph.set_inputs(offset=3)
            assert layout.violations(weights)["offscreen_area"] == 4
            objective = Objective().geometry(lambda: layout.offscreen_loss(weights))
            with graph.run(objective) as execution:
                execution.backward()
                assert weights["x"].grad == pytest.approx(4)
