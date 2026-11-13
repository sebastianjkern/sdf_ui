from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .autodiff import Parameter, Tensor

class RandomSource:
    def random(self) -> float: ...
    def uniform(self, low: float = ..., high: float = ...) -> float: ...
    def normal(self, mean: float = ..., std: float = ...) -> float: ...
    def integer(self, low: int, high: int | None = ...) -> int: ...
    def choice(self, values: Iterable[Any]) -> Any: ...

class GaussianPolicy:
    obs_dim: int
    action_dim: int
    hidden_dim: int
    w1: Parameter
    b1: Parameter
    w2: Parameter
    b2: Parameter
    @property
    def std(self) -> NDArray[np.float64]: ...
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = ...,
        seed: int = ...,
        initial_std: float | Sequence[float] = ...,
    ) -> None: ...
    def parameters(self) -> tuple[Parameter, ...]: ...
    def mean(self, observation: Any) -> Tensor: ...
    def mean_expressions(self, observation: Sequence[Any]) -> tuple[Any, ...]: ...
    def deterministic_action(self, observation: Any) -> NDArray[np.float64]: ...
    def log_prob(self, observation: Any, action: Any) -> Tensor: ...
    def sample(
        self, observation: Any, rng: Any
    ) -> tuple[NDArray[np.float64], Tensor]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

class CategoricalPolicy:
    obs_dim: int
    action_dim: int
    hidden_dim: int
    w1: Parameter
    b1: Parameter
    w2: Parameter
    b2: Parameter
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int = ..., seed: int = ...
    ) -> None: ...
    def parameters(self) -> tuple[Parameter, ...]: ...
    def logits(self, observation: Any) -> Tensor: ...
    def log_probs(self, observation: Any) -> Tensor: ...
    def probabilities(self, observation: Any) -> NDArray[np.float64]: ...
    def deterministic_action(self, observation: Any) -> int: ...
    def log_prob(self, observation: Any, action: int) -> Tensor: ...
    def sample(self, observation: Any, rng: Any) -> tuple[int, Tensor]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

class Transition:
    observation: NDArray[np.float64]
    action: NDArray[np.float64] | int
    log_prob: Tensor
    reward: float
    def __init__(
        self,
        observation: NDArray[np.float64],
        action: NDArray[np.float64] | int,
        log_prob: Tensor,
        reward: float,
    ) -> None: ...

class Episode:
    transitions: tuple[Transition, ...]
    policy_signature: Any
    def __init__(
        self, transitions: tuple[Transition, ...], policy_signature: Any
    ) -> None: ...
    @property
    def rewards(self) -> tuple[float, ...]: ...
    @property
    def total_reward(self) -> float: ...

def discounted_returns(
    rewards: Iterable[float], gamma: float = ...
) -> NDArray[np.float64]: ...

class PolicyGradientTrainer:
    policy: GaussianPolicy | CategoricalPolicy
    optimizer: Any
    rng: np.random.Generator
    gamma: float
    baseline: float
    updates: int
    last_metrics: dict[str, Any] | None
    @property
    def steps(self) -> int: ...
    def __init__(
        self,
        policy: GaussianPolicy | CategoricalPolicy,
        optimizer: Any = ...,
        seed: int = ...,
        gamma: float = ...,
    ) -> None: ...
    def collect_episode(
        self,
        rollout_fn: Callable[[Callable[[Any], Any], RandomSource], Iterable[float]],
    ) -> Episode: ...
    def collect(
        self, observation: Any, reward_fn: Callable[[Any, Any], float]
    ) -> Episode: ...
    def update(self, episodes: Iterable[Episode]) -> dict[str, Any]: ...
    def train(
        self,
        reward_fn: Callable[[Any, Any], float],
        observation_fn: Callable[[RandomSource], Any],
        steps: int = ...,
        batch_size: int = ...,
    ) -> list[dict[str, Any]]: ...
    def fit(
        self,
        environment: Any,
        observation_fn: Callable[[RandomSource], Mapping[str, float]],
        *,
        steps: int = ...,
        batch_size: int = ...,
        callback: Callable[[dict[str, Any]], Any] | None = ...,
        validation: Callable[[], float | Mapping[str, Any]] | None = ...,
        patience: int | None = ...,
        min_delta: float = ...,
        restore_best: bool = ...,
        validation_interval: int = ...,
    ) -> list[dict[str, Any]]: ...
    def evaluate(
        self, environment: Any, observations: Iterable[Mapping[str, float]]
    ) -> dict[str, Any]: ...
    def train_episodes(
        self,
        rollout_fn: Callable[[Callable[[Any], Any], RandomSource], Iterable[float]],
        steps: int = ...,
        batch_size: int = ...,
    ) -> list[dict[str, Any]]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
    def save(self, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> None: ...
