"""End-to-end learning workflows and constraint behavior."""



import numpy as np
import pytest

from sdf_ui import Canvas, param, sdf
from sdf_ui.graph import GraphCase, GraphDatasetTrainer, GraphTrainer, RenderEnvironment
from sdf_ui.learning import (
    SGD,
    Bounds,
    Element,
    GaussianPolicy,
    Layout,
    Objective,
    Parameter,
    ParameterStore,
    PolicyGradientTrainer,
)

_FIELD_TEXT = 'field'

_RADIUS_TEXT = 'radius'

def test_containment_and_content_overflow_have_useful_gradients():
    x = Parameter("x", -2.0)
    layout = Layout(
        [
            Element("parent", Bounds(0, 0, 10, 10)),
            Element(
                "child",
                Bounds(x, 1, 4, 4),
                parent="parent",
                content_bounds=Bounds(x - 1, 1, 4, 4),
            ),
        ],
        (20, 20),
    )
    loss = layout.containment_loss() + layout.content_overflow_loss()
    assert loss.item() == pytest.approx(5)
    loss.backward()
    assert x.grad == pytest.approx(-4)
    assert layout.violations()["containment_loss"] == pytest.approx(4)
    assert layout.violations()["content_overflow_loss"] == pytest.approx(1)


def test_weighted_pixel_mask_controls_loss_and_gradient():
    with Canvas((4, 3)) as canvas:
        p = ParameterStore([Parameter(_RADIUS_TEXT, 1.2)])
        mask = np.zeros((3, 4), dtype=np.float32)
        mask[:, :2] = 1
        with canvas.compile({_FIELD_TEXT: sdf.circle((2, 1), param(_RADIUS_TEXT))}, p) as graph:
            objective = Objective().mse(_FIELD_TEXT, 0, mask=mask, name="masked")
            with graph.run(objective) as execution:
                field = execution.read(_FIELD_TEXT)
                execution.backward()
                assert execution.loss == pytest.approx(
                    np.mean(field**2 * mask), abs=1e-6
                )
                assert p[_RADIUS_TEXT].grad == pytest.approx(
                    np.mean(-2 * field * mask), abs=1e-6
                )
            objective.set_mask("masked", 0)
            with graph.run(objective) as execution:
                assert execution.loss == 0
    with pytest.raises(ValueError, match="mask"):
        Objective().mse(_FIELD_TEXT, 0, mask=-1)


def test_rgba_mask_broadcasts_over_channels():
    with Canvas((4, 3)) as canvas:
        p = ParameterStore([Parameter(_RADIUS_TEXT, 1.2)])
        image = sdf.circle((2, 1), param(_RADIUS_TEXT)).fill("#808080", "#000000").to_rgb()
        mask = np.zeros((3, 4), dtype=np.float32)
        mask[1, 2] = 0.5
        with canvas.compile({"image": image}, p) as graph:
            with graph.run(Objective().mse("image", 0, mask=mask)) as execution:
                pixels = execution.read("image")
                execution.backward()
                expected = 2 * pixels * mask[..., None] / pixels.size
                np.testing.assert_allclose(
                    execution.gradient("image"), expected, atol=1e-7
                )
                assert execution.loss == pytest.approx(
                    np.mean(pixels**2 * mask[..., None]), abs=1e-7
                )


def test_dataset_batches_accumulate_weighted_gradients_and_evaluate_without_updates():
    with Canvas((4, 3)) as canvas:
        p = ParameterStore([Parameter(_RADIUS_TEXT, 1.2)])
        with (
            canvas.compile({_FIELD_TEXT: sdf.circle((1, 1), param(_RADIUS_TEXT))}, p) as a,
            canvas.compile(
                {_FIELD_TEXT: sdf.circle((3, 1), param(_RADIUS_TEXT))}, p, size=(6, 5)
            ) as b,
        ):
            cases = [
                GraphCase(a, Objective().mse(_FIELD_TEXT, 0), "a", 1),
                GraphCase(b, Objective().mse(_FIELD_TEXT, 1), "b", 3),
            ]
            individual = [_case_parameter_gradient(p, case) for case in cases]
            expected = (individual[0] + 3 * individual[1]) / 4
            optimizer = SGD(list(p.values()), lr=0.1)
            _assert_dataset_trainer(p, cases, optimizer, expected)


def _case_parameter_gradient(parameters, case):
    parameters[_RADIUS_TEXT].zero_grad()
    with case.graph.run(case.objective) as execution:
        pixels = execution.read(_FIELD_TEXT)
        execution.backward()
        gradient = float(parameters[_RADIUS_TEXT].grad)
        target = 0 if case.name == "a" else 1
        assert gradient == pytest.approx(np.mean(-2 * (pixels - target)), abs=1e-6)
    return gradient


def _assert_dataset_trainer(parameters, cases, optimizer, expected):
    with GraphDatasetTrainer(cases, optimizer) as trainer:
        before = parameters[_RADIUS_TEXT].item()
        metrics = trainer.evaluate()
        expected_loss = (metrics["cases"]["a"] + 3 * metrics["cases"]["b"]) / 4
        assert metrics["loss"] == pytest.approx(expected_loss)
        assert parameters[_RADIUS_TEXT].item() == before
        trainer.step()
        assert parameters[_RADIUS_TEXT].item() == pytest.approx(
            before - 1 / 10 * expected, abs=1e-6
        )
        _assert_dataset_checkpoint(trainer, parameters, cases)


def _assert_dataset_checkpoint(trainer, parameters, cases):
    checkpoint = trainer.state_dict()
    saved = parameters[_RADIUS_TEXT].item()
    trainer.step()
    trainer.load_state_dict(checkpoint)
    assert parameters[_RADIUS_TEXT].item() == saved
    assert trainer.steps == 1
    cases[1].objective.set_weight(cases[1].objective.pixel_losses[0].name, 2)
    with pytest.raises(ValueError, match="cases differ"):
        trainer.load_state_dict(checkpoint)


def test_policy_can_train_against_rendered_objective():
    with Canvas((4, 3)) as canvas:
        p = ParameterStore([Parameter(_RADIUS_TEXT, 1.0)])
        with canvas.compile({_FIELD_TEXT: sdf.circle((2, 1), param(_RADIUS_TEXT))}, p) as graph:
            environment = RenderEnvironment(
                graph, [_RADIUS_TEXT], objective=Objective().mse(_FIELD_TEXT, 0)
            )
            policy = GaussianPolicy(1, 1, seed=3, initial_std=0.5)
            trainer = PolicyGradientTrainer(policy, seed=4)
            initial = environment([1], policy.deterministic_action([1]))
            trainer.train(environment, lambda rng: [1], steps=15, batch_size=12)
            final = environment([1], policy.deterministic_action([1]))
            assert final > initial

            discrete = RenderEnvironment(
                graph,
                [_RADIUS_TEXT],
                objective=Objective().mse(_FIELD_TEXT, 0),
                action_transform=lambda choice: [1.0 + choice],
                reward=lambda obs, choice, run: -run.loss + choice,
            )
            reward = discrete([1], 1)
            assert np.isfinite(reward)
            assert p[_RADIUS_TEXT].item() == 2


def test_alternate_viewport_capture_and_execution():
    with Canvas((4, 3)) as canvas:
        p = ParameterStore([Parameter(_RADIUS_TEXT, 1.0)])
        with canvas.capture(sdf.circle((3, 2), 2), size=(6, 5)) as target:
            with canvas.compile(
                {_FIELD_TEXT: sdf.circle((3, 2), param(_RADIUS_TEXT))}, p, size=(6, 5)
            ) as graph:
                with graph.run(Objective().mse(_FIELD_TEXT, target)) as execution:
                    assert execution.read(_FIELD_TEXT).shape == (5, 6)
                    execution.backward()
                    assert p[_RADIUS_TEXT].grad == pytest.approx(-2, abs=1e-6)
                assert graph.render(backend="reference")[_FIELD_TEXT].shape == (5, 6)
                with pytest.raises(ValueError, match="Display rendering"):
                    graph.render()


def test_render_loss_backpropagates_to_policy_mean_weights():
    policy = GaussianPolicy(obs_dim=1, action_dim=1, hidden_dim=4, seed=7)
    observation = [0.7]
    radius = policy.mean_expressions(observation)[0].softplus()
    expected_loss = (policy.mean(observation)[0].softplus() - 2) ** 2
    expected_loss.backward()
    expected = {p.name: p.grad.copy() for p in policy.parameters()}
    for p in policy.parameters():
        p.zero_grad()
    with Canvas((4, 3)) as canvas:
        with canvas.capture(sdf.circle((2, 1), 2)) as target:
            with canvas.compile({_FIELD_TEXT: sdf.circle((2, 1), radius)}) as graph:
                objective = Objective().mse(_FIELD_TEXT, target)
                with graph.run(objective) as execution:
                    assert execution.loss == pytest.approx(
                        expected_loss.item(), abs=1e-6
                    )
                    gradients = execution.backward()
                    for name, gradient in expected.items():
                        np.testing.assert_allclose(gradients[name], gradient, atol=1e-6)
                with GraphTrainer(graph, objective, lr=0.05) as trainer:
                    initial = trainer.step()["loss"]
                    final = trainer.fit(15)[-1]["loss"]
                    assert final < initial * 1 / 5
