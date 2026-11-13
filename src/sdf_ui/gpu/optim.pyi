from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Self

from ..learning.autodiff import Parameter

class GPUAdam:
    canvas: Any
    parameters: tuple[Parameter, ...]
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    def __init__(
        self,
        canvas: Any,
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
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *_: Any) -> None: ...

class GPUSGD(GPUAdam):
    momentum: float
    nesterov: bool
    def __init__(
        self,
        canvas: Any,
        parameters: Iterable[Parameter] | Mapping[str, Parameter],
        lr: float = ...,
        momentum: float = ...,
        weight_decay: float = ...,
        nesterov: bool = ...,
    ) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
