from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from ..learning.objective import Objective
from .runtime import GraphExecution, RenderGraph

class RenderEnvironment:
    graph: RenderGraph
    actions: tuple[str, ...]
    observations: tuple[str, ...]
    objective: Objective | None
    reward: Callable[..., float] | None
    action_transform: Callable[..., Any] | None
    def __init__(
        self,
        graph: RenderGraph,
        actions: Sequence[str],
        *,
        observations: Sequence[str] = ...,
        objective: Objective | None = ...,
        reward: Callable[[Any, Any, GraphExecution], float] | None = ...,
        action_transform: Callable[[Any], Any] | None = ...,
    ) -> None: ...
    def __call__(
        self, observation: Sequence[float], action: Sequence[float]
    ) -> float: ...

class NamedValues(Mapping[str, float]):
    def __init__(self, values: Mapping[str, float]) -> None: ...
    def __getitem__(self, name: str) -> float: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def as_dict(self) -> dict[str, float]: ...

class NamedRenderEnvironment:
    graph: RenderGraph
    actions: Mapping[str, Any]
    observations: Mapping[str, Any]
    identity: str | None
    objective: Objective | None
    reward: Callable[..., float] | None
    action_transform: Callable[..., Any] | None
    def __init__(
        self,
        graph: RenderGraph,
        *,
        actions: Mapping[str, Any],
        observations: Mapping[str, Any] | None = ...,
        objective: Objective | None = ...,
        reward: Callable[[NamedValues, NamedValues, GraphExecution], float]
        | None = ...,
        action_transform: Callable[[Any], Any] | None = ...,
        identity: Any = ...,
    ) -> None: ...
    def schema(self) -> dict[str, Any]: ...
    def encode_observation(self, observation: Mapping[str, float]) -> list[float]: ...
    def decode_action(self, action: Any) -> dict[str, float]: ...
    def reward_vector(self, observation: Sequence[float], action: Any) -> float: ...
    def act(self, policy: Any, observation: Mapping[str, float]) -> NamedValues: ...
    def __call__(
        self, observation: Mapping[str, float], action: Mapping[str, float]
    ) -> float: ...
