"""Reward learning works without any renderer or reward derivative."""

import json
from copy import deepcopy

import numpy as np
import pytest

from sdf_ui.learning.autodiff import Tensor
from sdf_ui.learning.optimizers import SGD, Adam
from sdf_ui.learning.policy import (
    CategoricalPolicy,
    GaussianPolicy,
    PolicyGradientTrainer,
    discounted_returns,
)


def test_gaussian_sample_detaches_action_and_has_correct_score_gradient():
    policy = GaussianPolicy(2, 2, seed=3, initial_std=[0.2, 0.4])
    observation = np.array([0.3, -0.7])
    mean = policy.deterministic_action(observation)
    action, log_prob = policy.sample(observation, np.random.default_rng(8))
    saved_action = action.copy()
    assert type(action) is np.ndarray
    assert isinstance(log_prob, Tensor) and log_prob.shape == ()
    z = (saved_action - mean) / policy.std
    expected = np.sum(-0.5 * z**2 - np.log(policy.std * np.sqrt(2 * np.pi)))
    assert log_prob.item() == pytest.approx(expected)
    # Caller mutation must not alter the action stored in the score graph.
    action[:] = 999
    log_prob.backward()
    np.testing.assert_allclose(policy.b2.grad, (saved_action - mean) / policy.std**2)
    assert all(parameter.grad is not None for parameter in policy.parameters())


@pytest.mark.parametrize("policy_type", [GaussianPolicy, CategoricalPolicy])
def test_seeded_sampling_and_deterministic_actions(policy_type):
    first = policy_type(2, 3, seed=17)
    second = policy_type(2, 3, seed=17)
    rng1, rng2 = np.random.default_rng(13), np.random.default_rng(13)
    observation = [0.2, -0.8]
    expected = first.deterministic_action(observation)
    for _ in range(5):
        action1, logp1 = first.sample(observation, rng1)
        action2, logp2 = second.sample(observation, rng2)
        np.testing.assert_array_equal(action1, action2)
        assert logp1.item() == logp2.item()
    np.testing.assert_array_equal(first.deterministic_action(observation), expected)


def test_categorical_stable_log_softmax_and_score_gradient():
    policy = CategoricalPolicy(1, 3)
    for parameter in policy.parameters():
        parameter.assign(np.zeros(parameter.shape))
    policy.b2.assign([10000, 10001, 9999])
    probabilities = policy.probabilities([0])
    expected = np.exp([-1, 0, -2])
    expected /= expected.sum()
    np.testing.assert_allclose(probabilities, expected)
    assert policy.deterministic_action([0]) == 1
    score = policy.log_prob([0], 2)
    score.backward()
    np.testing.assert_allclose(policy.b2.grad, np.array([0, 0, 1]) - expected)
    assert np.all(np.isfinite(policy.log_probs([0]).numpy()))
    with pytest.raises(ValueError, match="range"):
        policy.log_prob([0], 3)
    with pytest.raises(ValueError, match="integer"):
        policy.log_prob([0], 0.5)


def _target_centers(observation):
    width, height = observation
    return np.array(
        [
            0.23 + 0.12 * width,
            0.5 + 0.15 * height,
            0.77 - 0.10 * width,
            0.5 - 0.12 * height,
        ]
    )


def _blackbox_layout_reward(observation, action):
    """Quantized positions and overlap penalties are deliberately NumPy only."""
    assert type(action) is np.ndarray
    # A pixel-grid measurement: the score cannot backpropagate through rounding.
    centers = np.rint(action * 64) / 64
    target = _target_centers(observation)
    squared_error = float(np.sum((centers - target) ** 2))
    separation = np.linalg.norm(centers[:2] - centers[2:])
    overlap = max(0, 0.25 - separation) ** 2
    outside = np.maximum(0, -centers) + np.maximum(0, centers - 1)
    return float(-squared_error - 2 * overlap - np.sum(outside**2))


@pytest.mark.parametrize("seed", [0, 7])
def test_gaussian_improves_contextual_button_placement_blackbox_reward(seed):
    policy = GaussianPolicy(2, 4, hidden_dim=12, seed=seed, initial_std=0.12)
    trainer = PolicyGradientTrainer(policy, Adam(policy.parameters(), lr=0.015), seed)
    contexts = np.random.default_rng(123).uniform(-1, 1, (40, 2))

    def evaluate():
        return np.mean(
            [
                _blackbox_layout_reward(obs, policy.deterministic_action(obs))
                for obs in contexts
            ]
        )

    before = evaluate()
    history = trainer.train(
        _blackbox_layout_reward,
        lambda rng: rng.uniform(-1, 1, 2),
        steps=90,
        batch_size=32,
    )
    after = evaluate()
    assert after > before + 0.5
    assert after > -0.03
    assert np.mean([step["mean_reward"] for step in history[-10:]]) > np.mean(
        [step["mean_reward"] for step in history[:10]]
    )
    assert len(history) == 90
    assert trainer.updates == 90
    np.testing.assert_array_equal(policy.std, np.full(4, 0.12))


def test_categorical_improves_contextual_row_column_grid_selection():
    policy = CategoricalPolicy(3, 3, seed=5)
    trainer = PolicyGradientTrainer(policy, seed=21)
    contexts = np.eye(3)

    def observation_fn(rng):
        return contexts[rng.integers(3)]

    def reward_fn(observation, action):
        assert isinstance(action, int)
        return 1.0 if action == int(np.argmax(observation)) else -1.0

    before = np.mean(
        [policy.probabilities(obs)[index] for index, obs in enumerate(contexts)]
    )
    trainer.train(reward_fn, observation_fn, steps=65, batch_size=24)
    after = np.mean(
        [policy.probabilities(obs)[index] for index, obs in enumerate(contexts)]
    )
    assert after > before + 0.5
    assert after > 0.95
    assert [policy.deterministic_action(obs) for obs in contexts] == [0, 1, 2]


def test_discounted_returns_and_update_use_terminal_reward_credit():
    np.testing.assert_allclose(discounted_returns([0, 0, 8], 0.5), [2, 4, 8])
    np.testing.assert_array_equal(discounted_returns([1, 2, 3], 0), [1, 2, 3])
    np.testing.assert_array_equal(discounted_returns([1, 2, 3], 1), [6, 5, 3])
    policy = GaussianPolicy(1, 1, initial_std=0.3)
    optimizer = SGD(policy.parameters(), lr=0.02)
    trainer = PolicyGradientTrainer(policy, optimizer, seed=9, gamma=0.5)

    def rollout(act, rng):
        for observation in [[-1], [0], [1]]:
            act(observation)
        return [0, 0, 8]

    episode = trainer.collect_episode(rollout)
    assert episode.rewards == (0, 0, 8)
    assert episode.total_reward == 8
    returns = np.array([2, 4, 8])
    advantages = (returns - returns.mean()) / returns.std()
    expected_gradient = -np.mean(
        [
            (
                (
                    transition.action
                    - policy.deterministic_action(transition.observation)
                )
                / policy.std**2
            )
            * advantage
            for transition, advantage in zip(episode.transitions, advantages)
        ],
        axis=0,
    )
    old_bias = policy.b2.numpy()
    metrics = trainer.update([episode])
    np.testing.assert_allclose(policy.b2.numpy(), old_bias - 0.02 * expected_gradient)
    assert metrics["mean_return"] == pytest.approx(14 / 3)
    assert metrics["mean_reward"] == 8
    assert metrics["transitions"] == 3
    with pytest.raises(ValueError, match="current policy"):
        trainer.update([episode])


def test_episodic_training_learns_from_delayed_blackbox_reward():
    policy = CategoricalPolicy(2, 2, seed=6)
    trainer = PolicyGradientTrainer(policy, seed=10, gamma=0.95)
    contexts = np.eye(2)

    def rollout(act, rng):
        first = act(contexts[0])
        second = act(contexts[1])
        # Each choice is scored only after both placements have been made.
        return [0, float((first == 0) + (second == 1))]

    trainer.train_episodes(rollout, steps=65, batch_size=16)
    assert policy.probabilities(contexts[0])[0] > 0.9
    assert policy.probabilities(contexts[1])[1] > 0.9


@pytest.mark.parametrize("policy_type", [GaussianPolicy, CategoricalPolicy])
@pytest.mark.parametrize("optimizer_type", [Adam, SGD])
def test_json_checkpoint_resumes_rng_optimizer_and_training_exactly(
    policy_type,
    optimizer_type,
):
    def reward_fn(obs, action):
        if isinstance(action, int):
            return float(action == int(obs[0] > 0))
        return -float(np.sum((action - obs[0]) ** 2))

    def observation_fn(rng):
        return rng.uniform(-1, 1, 2)

    policy = policy_type(2, 2, seed=3)
    optimizer = optimizer_type(policy.parameters(), lr=0.02)
    trainer = PolicyGradientTrainer(policy, optimizer, seed=10, gamma=0.8)
    trainer.train(reward_fn, observation_fn, steps=3, batch_size=5)
    checkpoint = json.loads(json.dumps(trainer.state_dict(), allow_nan=False))
    restored_policy = policy_type(2, 2, seed=99)
    restored = PolicyGradientTrainer(
        restored_policy, optimizer_type(restored_policy.parameters(), lr=0.001), seed=99
    )
    restored.load_state_dict(checkpoint)
    expected = trainer.train(reward_fn, observation_fn, steps=3, batch_size=5)
    actual = restored.train(reward_fn, observation_fn, steps=3, batch_size=5)
    assert actual == expected
    assert restored.state_dict() == trainer.state_dict()


def test_policy_checkpoint_validation_is_atomic_and_copies_state():
    policy = GaussianPolicy(2, 2, initial_std=[0.1, 0.3])
    saved = policy.state_dict()
    wrong_shape = deepcopy(saved)
    wrong_shape["weights"]["b2"] = [1]
    with pytest.raises(ValueError, match="shape"):
        policy.load_state_dict(wrong_shape)
    assert policy.state_dict() == saved
    wrong_std = deepcopy(saved)
    wrong_std["std"] = [0, 0.3]
    with pytest.raises(ValueError, match="positive"):
        policy.load_state_dict(wrong_std)
    assert policy.state_dict() == saved
    wrong_value = deepcopy(saved)
    wrong_value["weights"]["b2"] = [np.nan, 0]
    with pytest.raises(ValueError, match="finite"):
        policy.load_state_dict(wrong_value)
    assert policy.state_dict() == saved
    saved["weights"]["b2"][0] = 100
    assert policy.b2.numpy()[0] == 0
    loaded = GaussianPolicy(2, 2, seed=99, initial_std=0.5)
    loaded.load_state_dict(policy.state_dict())
    np.testing.assert_array_equal(loaded.std, policy.std)
    np.testing.assert_array_equal(
        loaded.deterministic_action([0.2, 0.4]),
        policy.deterministic_action([0.2, 0.4]),
    )


@pytest.mark.parametrize("policy_type", [GaussianPolicy, CategoricalPolicy])
@pytest.mark.parametrize("observation", [[1], [[1, 2]], [np.inf, 0], [0, np.nan]])
def test_invalid_observations_rejected(policy_type, observation):
    policy = policy_type(2, 2)
    with pytest.raises(ValueError, match="shape|finite"):
        policy.sample(observation, np.random.default_rng(0))
    with pytest.raises(ValueError, match="shape|finite"):
        policy.deterministic_action(observation)


@pytest.mark.parametrize("std", [0, -0.1, np.nan, np.inf, [0.1], [[0.1, 0.2]]])
def test_invalid_std_rejected(std):
    with pytest.raises(ValueError, match="std"):
        GaussianPolicy(2, 2, initial_std=std)


def test_bad_rollouts_and_foreign_or_changed_policy_rejected():
    policy = GaussianPolicy(1, 1)
    trainer = PolicyGradientTrainer(policy)
    with pytest.raises(ValueError, match="at least one action"):
        trainer.collect_episode(lambda act, rng: [])
    with pytest.raises(ValueError, match="one reward per action"):
        trainer.collect_episode(lambda act, rng: (act([0]), [])[1])
    with pytest.raises(ValueError, match="finite"):
        trainer.collect([0], lambda obs, action: np.nan)
    with pytest.raises(ValueError, match="shape"):
        trainer.collect([0], lambda obs, action: [1])
    episode = trainer.collect([0], lambda obs, action: 1)
    other = PolicyGradientTrainer(GaussianPolicy(1, 1))
    with pytest.raises(ValueError, match="current policy"):
        other.update([episode])
    policy.b2.assign([1])
    with pytest.raises(ValueError, match="current policy"):
        trainer.update([episode])
    with pytest.raises(ValueError, match="at least one episode"):
        trainer.update([])


def test_single_transition_keeps_gradient_and_constant_batch_is_finite():
    policy = GaussianPolicy(1, 1)
    trainer = PolicyGradientTrainer(policy, seed=4)
    before = policy.b2.numpy()
    trainer.update([trainer.collect([0], lambda obs, action: 2)])
    assert not np.array_equal(before, policy.b2.numpy())
    metrics = trainer.update(
        [trainer.collect([0], lambda obs, action: 2) for _ in range(4)]
    )
    assert metrics["advantage_std"] == 0
    assert np.isfinite(metrics["loss"])


@pytest.mark.parametrize("optimizer_type", [Adam, SGD])
def test_optimizer_must_own_exact_policy_parameters(optimizer_type):
    policy = GaussianPolicy(1, 1)
    wrong = GaussianPolicy(1, 1)
    with pytest.raises(ValueError, match="exactly this policy"):
        PolicyGradientTrainer(policy, optimizer_type(wrong.parameters()))
    with pytest.raises(ValueError, match="exactly this policy"):
        PolicyGradientTrainer(policy, optimizer_type(policy.parameters()[:2]))


@pytest.mark.parametrize("gamma", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_discount_rejected(gamma):
    with pytest.raises(ValueError, match="gamma"):
        PolicyGradientTrainer(GaussianPolicy(1, 1), gamma=gamma)
    with pytest.raises(ValueError, match="gamma"):
        discounted_returns([1], gamma)


def test_training_loop_validates_counts_and_accepts_zero_steps():
    trainer = PolicyGradientTrainer(GaussianPolicy(1, 1))
    reward, observation = lambda obs, action: 1, lambda rng: [0]
    assert trainer.train(reward, observation, steps=0) == []
    for kwargs in ({"steps": -1}, {"steps": 1.5}, {"batch_size": 0}):
        with pytest.raises(ValueError, match="integer"):
            trainer.train(reward, observation, **kwargs)
