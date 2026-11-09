"""NumPy policies trained from scalar rewards, including black-box render scores.

``trainer.train(reward_fn, observation_fn, steps=100, batch_size=32)`` is the
single-step interface: ``observation_fn(rng)`` supplies a context and
``reward_fn(observation, action)`` scores a sampled layout. No gradient through
the reward or renderer is required. For an episode, ``rollout_fn(act, rng)``
calls ``act(observation)`` at each step and returns one scalar reward per call.
Use ``collect_episode`` and ``update`` to batch trajectories, or
``train_episodes`` for the complete loop. Updates consume on-policy episodes;
collect fresh episodes after each update.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np

from .autodiff import SGD, Adam, Parameter, Tensor, as_tensor, stack

__all__ = [
    "CategoricalPolicy",
    "Episode",
    "GaussianPolicy",
    "PolicyGradientTrainer",
    "RandomSource",
    "Transition",
    "discounted_returns",
]


class RandomSource:
    """Seeded scalar random values for high-level observation callbacks."""

    def __init__(self, generator):
        self._generator = generator

    def random(self):
        return float(self._generator.random())

    def uniform(self, low=0.0, high=1.0):
        return float(self._generator.uniform(low, high))

    def normal(self, mean=0.0, std=1.0):
        return float(self._generator.normal(mean, std))

    def integer(self, low, high=None):
        return int(self._generator.integers(low, high))

    def choice(self, values):
        values = tuple(values)
        if not values:
            raise ValueError("choice requires at least one value")
        return values[self.integer(len(values))]


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _array(value, shape, name):
    if isinstance(value, Tensor):
        value = value.numpy()
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _scalar(value, name):
    return float(_array(value, (), name))


def _json_value(value):
    """Copy optimizer/RNG state into plain JSON types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return deepcopy(value)


class _MLP:
    """One tanh hidden layer, with a linear output head."""

    def __init__(self, obs_dim, action_dim, hidden_dim=16, seed=0):
        self.obs_dim = _integer(obs_dim, "obs_dim")
        self.action_dim = _integer(action_dim, "action_dim")
        self.hidden_dim = _integer(hidden_dim, "hidden_dim")
        rng = np.random.default_rng(seed)
        self.w1 = Parameter(
            "w1",
            rng.normal(0, 1 / np.sqrt(self.obs_dim), (self.obs_dim, self.hidden_dim)),
        )
        self.b1 = Parameter("b1", np.zeros(self.hidden_dim))
        self.w2 = Parameter(
            "w2",
            rng.normal(
                0, 0.1 / np.sqrt(self.hidden_dim), (self.hidden_dim, self.action_dim)
            ),
        )
        self.b2 = Parameter("b2", np.zeros(self.action_dim))

    def parameters(self):
        """Return the trainable weights, in stable optimizer order."""
        return (self.w1, self.b1, self.w2, self.b2)

    def _forward(self, observation):
        obs = _array(observation, (self.obs_dim,), "observation")
        result = (as_tensor(obs) @ self.w1 + self.b1).tanh() @ self.w2 + self.b2
        _array(result, (self.action_dim,), "policy output")
        return result

    def _signature(self):
        return (id(self), tuple(parameter.version for parameter in self.parameters()))

    def state_dict(self):
        """Return a JSON-serializable checkpoint, independent of live arrays."""
        return {
            "type": type(self).__name__,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "weights": {p.name: p.numpy().tolist() for p in self.parameters()},
        }

    def _validated_state(self, state):
        expected = (type(self).__name__, self.obs_dim, self.action_dim, self.hidden_dim)
        actual = tuple(
            state.get(k) for k in ("type", "obs_dim", "action_dim", "hidden_dim")
        )
        if actual != expected:
            raise ValueError("Checkpoint policy type or dimensions do not match")
        weights = state.get("weights", {})
        if set(weights) != {p.name for p in self.parameters()}:
            raise ValueError("Checkpoint weight names do not match")
        return {
            p.name: _array(weights[p.name], p.shape, p.name) for p in self.parameters()
        }

    def load_state_dict(self, state):
        """Validate all weights before assigning; preserve Parameter identities."""
        values = self._validated_state(state)
        for parameter in self.parameters():
            parameter.assign(values[parameter.name])


class GaussianPolicy(_MLP):
    """An MLP mean with independent Gaussian exploration in each action axis.

    Actions are unconstrained vectors. Use normalized coordinates and score
    bounds/overlap in the reward, or transform actions inside the environment.
    ``initial_std`` is a positive scalar or vector of length ``action_dim``.
    Exploration remains fixed during training, preventing variance collapse.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=16, seed=0, initial_std=0.25):
        super().__init__(obs_dim, action_dim, hidden_dim, seed)
        self._std = self._validate_std(initial_std)

    def _validate_std(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape == ():
            value = np.full(self.action_dim, float(value))
        value = _array(value, (self.action_dim,), "std")
        if np.any(value <= 0):
            raise ValueError("std must be positive")
        return value

    @property
    def std(self):
        return self._std.copy()

    def mean(self, observation):
        return self._forward(observation)

    def mean_expressions(self, observation):
        """Return lazy scalar expressions for the deterministic policy mean.

        Use these expressions as scene parameters to backpropagate render losses
        into policy weights. Observations can be fixed scalars or graph input
        expressions, and graphs can share the policy's Parameter leaves. This
        path optimizes the mean; it does not differentiate sampled actions.
        """
        from ..core.expressions import Expr

        obs = tuple(observation)
        if len(obs) != self.obs_dim:
            raise ValueError("observation must match policy obs_dim")
        obs = tuple(
            value if isinstance(value, Expr) else _scalar(value, "observation")
            for value in obs
        )

        def leaf(parameter, *index):
            return Expr("index", (parameter, index))

        hidden = [
            (
                sum(
                    (obs[i] * leaf(self.w1, i, j) for i in range(self.obs_dim)),
                    leaf(self.b1, j),
                )
            ).tanh()
            for j in range(self.hidden_dim)
        ]
        return tuple(
            sum(
                (hidden[j] * leaf(self.w2, j, k) for j in range(self.hidden_dim)),
                leaf(self.b2, k),
            )
            for k in range(self.action_dim)
        )

    def deterministic_action(self, observation):
        return self.mean(observation).numpy()

    def log_prob(self, observation, action):
        action = _array(action, (self.action_dim,), "action")
        return self._log_prob(self.mean(observation), action)

    def _log_prob(self, mean, action):
        z = (as_tensor(action) - mean) / self._std
        result = (-0.5 * z * z - np.log(self._std) - 0.5 * np.log(2 * np.pi)).sum()
        _scalar(result, "log probability")
        return result

    def sample(self, observation, rng):
        """Return a detached ndarray and differentiable scalar log probability.

        The score is evaluated at a constant action: retaining a derivative
        through sampling would cancel the REINFORCE mean gradient.
        """
        mean = self.mean(observation)
        action = _array(
            rng.normal(mean.numpy(), self._std), (self.action_dim,), "sampled action"
        )
        return action, self._log_prob(mean, action.copy())

    def state_dict(self):
        state = super().state_dict()
        state["std"] = self._std.tolist()
        return state

    def _validated_state(self, state):
        values = super()._validated_state(state)
        self._validate_std(state["std"])
        return values

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self._std = self._validate_std(state["std"])


class CategoricalPolicy(_MLP):
    """An MLP categorical distribution for discrete layout alternatives."""

    def logits(self, observation):
        return self._forward(observation)

    def log_probs(self, observation):
        logits = self.logits(observation)
        # The max is a detached constant. Its derivative cancels in log-softmax.
        shifted = logits - float(np.max(logits.numpy()))
        result = shifted - shifted.exp().sum().log()
        _array(result, (self.action_dim,), "log probabilities")
        return result

    def probabilities(self, observation):
        probabilities = np.exp(self.log_probs(observation).numpy())
        return probabilities / probabilities.sum()

    def deterministic_action(self, observation):
        return int(np.argmax(self.logits(observation).numpy()))

    def log_prob(self, observation, action):
        action = _integer(action, "action", minimum=0)
        if action >= self.action_dim:
            raise ValueError("action category is out of range")
        return self.log_probs(observation)[action]

    def sample(self, observation, rng):
        log_probs = self.log_probs(observation)
        probabilities = np.exp(log_probs.numpy())
        probabilities /= probabilities.sum()
        action = int(rng.choice(self.action_dim, p=probabilities))
        return action, log_probs[action]


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    action: Union[np.ndarray, int]
    log_prob: Tensor
    reward: float


@dataclass(frozen=True)
class Episode:
    transitions: Tuple[Transition, ...]
    policy_signature: Any

    @property
    def rewards(self):
        return tuple(transition.reward for transition in self.transitions)

    @property
    def total_reward(self):
        return float(sum(self.rewards))


def discounted_returns(rewards, gamma=0.99):
    """Compute reward-to-go without crossing episode boundaries."""
    gamma = _scalar(gamma, "gamma")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    rewards = np.asarray(list(rewards), dtype=float)
    if rewards.ndim != 1 or not np.all(np.isfinite(rewards)):
        raise ValueError("rewards must be a finite one-dimensional sequence")
    result = np.empty_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    if not np.all(np.isfinite(result)):
        raise ValueError("discounted returns must be finite")
    return result


class PolicyGradientTrainer:
    """Reward-to-go REINFORCE with a scalar baseline and standardized advantages.

    Batches with multiple transitions use their mean return as baseline and
    divide centered returns by their standard deviation when nonzero. A single
    transition instead uses the previous batch's mean baseline (initially 0),
    preserving a learning signal. Reward callbacks receive detached values.
    Default optimization uses Adam with learning rate 0.01; pass an explicit
    Adam/SGD instance to configure optimization.
    """

    def __init__(self, policy, optimizer=None, seed=0, gamma=0.99):
        if not isinstance(policy, (GaussianPolicy, CategoricalPolicy)):
            raise TypeError("policy must be GaussianPolicy or CategoricalPolicy")
        self.policy = policy
        self.optimizer = (
            optimizer if optimizer is not None else Adam(policy.parameters(), lr=0.01)
        )
        if isinstance(self.optimizer, (Adam, SGD)):
            expected = {id(parameter) for parameter in policy.parameters()}
            actual = tuple(self.optimizer.parameters)
            if len(actual) != len(expected) or {id(p) for p in actual} != expected:
                raise ValueError("Optimizer must own exactly this policy's parameters")
        self.rng = np.random.default_rng(seed)
        self.gamma = _scalar(gamma, "gamma")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        self.baseline = 0.0
        self.updates = 0
        self.last_metrics = None
        self._environment_schema = None

    @property
    def steps(self):
        return self.updates

    def collect_episode(self, rollout_fn):
        """Run ``rollout_fn(act, rng) -> rewards``, recording each act call.

        Rewards can be delayed (e.g. zeros followed by a final screenshot
        score). There must be exactly one finite scalar reward per action.
        The callback owns resetting and terminating its environment.
        """
        signature = self.policy._signature()
        records = []

        def act(observation):
            obs = _array(observation, (self.policy.obs_dim,), "observation")
            action, log_prob = self.policy.sample(obs, self.rng)
            saved_action = action.copy() if isinstance(action, np.ndarray) else action
            obs.setflags(write=False)
            if isinstance(saved_action, np.ndarray):
                saved_action.setflags(write=False)
            records.append((obs, saved_action, log_prob))
            return action

        rewards = list(rollout_fn(act, self.rng))
        if not records:
            raise ValueError("An episode must contain at least one action")
        if len(records) != len(rewards):
            raise ValueError("Expected exactly one reward per action")
        if signature != self.policy._signature():
            raise ValueError("Policy changed during rollout")
        transitions = tuple(
            Transition(obs, action, log_prob, _scalar(reward, "reward"))
            for (obs, action, log_prob), reward in zip(records, rewards)
        )
        return Episode(transitions, signature)

    def collect(self, observation, reward_fn):
        """Collect one contextual layout; ``reward_fn(obs, action) -> float``."""
        obs = _array(observation, (self.policy.obs_dim,), "observation")

        def rollout(act, rng):
            action = act(obs)
            return [reward_fn(obs.copy(), action)]

        return self.collect_episode(rollout)

    def update(self, episodes):
        """Consume a batch of fresh episodes and return scalar metrics."""
        episodes = list(episodes)
        if not episodes:
            raise ValueError("An update requires at least one episode")
        signature = self.policy._signature()
        returns = []
        log_probs = []
        for episode in episodes:
            if not isinstance(episode, Episode) or not episode.transitions:
                raise ValueError("Expected nonempty Episode objects")
            if episode.policy_signature != signature:
                raise ValueError("Episodes must come from the current policy weights")
            returns.extend(discounted_returns(episode.rewards, self.gamma))
            for transition in episode.transitions:
                _scalar(transition.log_prob, "log probability")
                log_probs.append(transition.log_prob)
        returns = np.asarray(returns)
        baseline = float(returns.mean()) if len(returns) > 1 else self.baseline
        advantages = returns - baseline
        scale = float(advantages.std())
        if scale > 1e-8:
            advantages /= scale
        loss = -(stack(log_probs) * advantages).mean()
        loss_value = _scalar(loss, "loss")
        self.optimizer.zero_grad()
        loss.backward()
        for parameter in self.policy.parameters():
            if parameter.grad is None or not np.all(np.isfinite(parameter.grad)):
                raise ValueError("Policy gradients must be connected and finite")
        self.optimizer.step()
        self.baseline = float(returns.mean())
        self.updates += 1
        result = {
            "loss": loss_value,
            "mean_reward": float(
                np.mean([episode.total_reward for episode in episodes])
            ),
            "mean_return": self.baseline,
            "baseline": baseline,
            "advantage_std": scale,
            "episodes": len(episodes),
            "transitions": len(returns),
            "update": self.updates,
            "step": self.updates,
        }
        self.last_metrics = result
        return result

    def train(self, reward_fn, observation_fn, steps=100, batch_size=32):
        """Train one-shot layouts and return one metrics dict per update."""
        steps = _integer(steps, "steps", minimum=0)
        batch_size = _integer(batch_size, "batch_size")
        history = []
        for _ in range(steps):
            episodes = [
                self.collect(observation_fn(self.rng), reward_fn)
                for _ in range(batch_size)
            ]
            history.append(self.update(episodes))
        return history

    def fit(
        self,
        environment,
        observation_fn,
        *,
        steps=100,
        batch_size=32,
        callback=None,
        validation=None,
        patience=None,
        min_delta=0.0,
        restore_best=True,
        validation_interval=1,
    ):
        """Train a named render environment with named observations."""
        if not callable(observation_fn):
            raise TypeError("observation_fn must be callable")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        if (
            not callable(getattr(environment, "encode_observation", None))
            or not callable(getattr(environment, "reward_vector", None))
            or not callable(getattr(environment, "schema", None))
        ):
            raise TypeError("fit requires a named render environment")
        if len(environment.observations) != self.policy.obs_dim:
            raise ValueError("Policy observation dimension differs from environment")
        if (
            isinstance(self.policy, CategoricalPolicy)
            and environment.action_transform is None
        ):
            raise ValueError(
                "Categorical policies require an action_transform returning named actions"
            )
        if (
            environment.action_transform is None
            and len(environment.actions) != self.policy.action_dim
        ):
            raise ValueError("Policy action dimension differs from environment")
        steps = _integer(steps, "steps", minimum=0)
        batch_size = _integer(batch_size, "batch_size")
        schema = environment.schema()
        if self._environment_schema is not None and self._environment_schema != schema:
            raise ValueError("Policy trainer environment schema differs")
        self._environment_schema = schema

        def step():
            episodes = [
                self.collect(
                    environment.encode_observation(
                        observation_fn(RandomSource(self.rng))
                    ),
                    environment.reward_vector,
                )
                for _ in range(batch_size)
            ]
            return self.update(episodes)

        from .training import fit_steps

        return fit_steps(
            self,
            steps,
            callback,
            validation=validation,
            patience=patience,
            min_delta=min_delta,
            restore_best=restore_best,
            validation_interval=validation_interval,
            step=step,
        )

    def evaluate(self, environment, observations):
        """Score deterministic actions on named held-out observations."""
        if (
            self._environment_schema is not None
            and environment.schema() != self._environment_schema
        ):
            raise ValueError("Policy trainer environment schema differs")
        rewards = []
        for observation in observations:
            vector = environment.encode_observation(observation)
            action = self.policy.deterministic_action(vector)
            rewards.append(environment.reward_vector(vector, action))
        if not rewards:
            raise ValueError("Evaluation requires at least one observation")
        return {"mean_reward": float(np.mean(rewards)), "rewards": rewards}

    def train_episodes(self, rollout_fn, steps=100, batch_size=8):
        """Train environment trajectories with reward-to-go credit assignment."""
        steps = _integer(steps, "steps", minimum=0)
        batch_size = _integer(batch_size, "batch_size")
        history = []
        for _ in range(steps):
            history.append(
                self.update(
                    [self.collect_episode(rollout_fn) for _ in range(batch_size)]
                )
            )
        return history

    def state_dict(self):
        """Save policy, RNG, baseline, counters, and supported optimizer state."""
        if (
            self._environment_schema is not None
            and self._environment_schema["identity"] is None
        ):
            raise ValueError(
                "Named policy checkpoints require an explicit environment identity"
            )
        save = getattr(self.optimizer, "state_dict", None)
        state = {
            "format_version": 2,
            "environment": deepcopy(self._environment_schema),
            "policy": self.policy.state_dict(),
            "gamma": self.gamma,
            "baseline": self.baseline,
            "updates": self.updates,
            "rng": deepcopy(self.rng.bit_generator.state),
            "optimizer_type": type(self.optimizer).__name__,
            "optimizer": save() if callable(save) else None,
        }
        return _json_value(state)

    def load_state_dict(self, state):
        """Restore a checkpoint; create the same policy and optimizer types first."""
        if (
            not isinstance(state, dict)
            or state.get("format_version") != 2
            or "environment" not in state
        ):
            raise ValueError("Invalid policy trainer checkpoint")
        environment = state.get("environment")
        if environment is not None and (
            not isinstance(environment, dict) or environment.get("identity") is None
        ):
            raise ValueError("Invalid policy environment schema")
        if (
            self._environment_schema is not None
            and environment != self._environment_schema
        ):
            raise ValueError("Policy trainer environment schema differs")
        self.policy._validated_state(state["policy"])
        gamma = _scalar(state["gamma"], "gamma")
        baseline = _scalar(state["baseline"], "baseline")
        updates = _integer(state["updates"], "updates", minimum=0)
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state["rng"])
        optimizer_state = state.get("optimizer")
        if optimizer_state is not None:
            load = getattr(self.optimizer, "load_state_dict", None)
            if state["optimizer_type"] != type(self.optimizer).__name__ or not callable(
                load
            ):
                raise ValueError("Checkpoint optimizer type is incompatible")
            # Optimizers validate their entire state before assigning it.
            load(deepcopy(optimizer_state))
        self.policy.load_state_dict(state["policy"])
        self.gamma, self.baseline, self.updates = gamma, baseline, updates
        self.rng = rng
        self._environment_schema = deepcopy(environment)
        self.last_metrics = None

    def save(self, path):
        """Write policy, optimizer, and random state to JSON."""
        Path(path).write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    def load(self, path):
        self.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
