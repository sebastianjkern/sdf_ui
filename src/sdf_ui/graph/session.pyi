from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Self

from ..learning.objective import Objective
from .runtime import GraphTrainer, RenderGraph

class TrainingSession:
    DEFAULT_OUTPUT: str
    canvas: Any
    graph: RenderGraph
    outputs: tuple[str, ...]
    objective: Objective
    trainer: GraphTrainer | None
    closed: bool
    def __init__(
        self,
        canvas: Any,
        outputs: Any,
        params: Any = ...,
        *,
        size: tuple[int, int] | None = ...,
        inputs: Mapping[str, Any] | None = ...,
    ) -> None: ...
    def match(
        self,
        target: Any,
        *,
        output: str | None = ...,
        kind: str = ...,
        weight: float = ...,
        name: str | None = ...,
        mask: Any = ...,
        normalization: str = ...,
        delta: float = ...,
    ) -> Self: ...
    def geometry(self, callback: Callable[..., Any], **options: Any) -> Self: ...
    def fit(
        self,
        steps: int,
        *,
        lr: float | None = ...,
        callback: Callable[[dict[str, Any]], Any] | None = ...,
        validation: Callable[[], float | Mapping[str, Any]] | None = ...,
        patience: int | None = ...,
        min_delta: float = ...,
        restore_best: bool = ...,
        validation_interval: int = ...,
    ) -> list[dict[str, Any]]: ...
    def evaluate(self) -> dict[str, Any]: ...
    def report(self, *, layout: Any = ...) -> dict[str, Any]: ...
    def save(self, path: str | Path, *, output: str | None = ...) -> Any: ...
    def save_checkpoint(self, path: str | Path) -> None: ...
    def load_checkpoint(self, path: str | Path, *, lr: float | None = ...) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *_: Any) -> None: ...
