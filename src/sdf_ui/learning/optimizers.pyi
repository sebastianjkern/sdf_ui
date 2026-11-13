from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .autodiff import Parameter

class SGD:
    parameters: tuple[Parameter, ...]
    lr: float
    momentum: float
    weight_decay: float
    nesterov: bool
    def __init__(
        self,
        parameters: Iterable[Parameter] | Mapping[str, Parameter],
        lr: float = ...,
        momentum: float = ...,
        weight_decay: float = ...,
        nesterov: bool = ...,
    ) -> None: ...
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

class Adam:
    parameters: tuple[Parameter, ...]
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    def __init__(
        self,
        parameters: Iterable[Parameter] | Mapping[str, Parameter],
        lr: float = ...,
        betas: tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
    ) -> None: ...
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
