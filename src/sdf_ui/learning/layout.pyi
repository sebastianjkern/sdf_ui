from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ..core.sdf import SDFTexture
from .autodiff import Parameter, Tensor

class Bounds:
    x: Any
    y: Any
    width: Any
    height: Any
    @property
    def left(self) -> Any: ...
    @property
    def right(self) -> Any: ...
    @property
    def top(self) -> Any: ...
    @property
    def bottom(self) -> Any: ...
    @property
    def center_x(self) -> Any: ...
    @property
    def center_y(self) -> Any: ...
    @property
    def center(self) -> tuple[Any, Any]: ...
    @property
    def area(self) -> Any: ...
    def __init__(self, x: Any, y: Any, width: Any, height: Any) -> None: ...

class Element:
    name: str
    bounds: Bounds
    role: str
    parent: str | None
    minimum_size: tuple[Any, Any] | None
    content_bounds: Bounds | None
    def __init__(
        self,
        name: str,
        bounds: Bounds,
        role: str = ...,
        parent: Element | str | None = ...,
        minimum_size: tuple[Any, Any] | None = ...,
        content_bounds: Bounds | None = ...,
    ) -> None: ...
    def shape(self, corner_radius: Any = ...) -> SDFTexture: ...
    def to_sdf(self, corner_radius: Any = ...) -> SDFTexture: ...

class LayoutSnapshot:
    elements: Mapping[str, Element]
    bounds: Mapping[str, Bounds]
    size: tuple[float, float]
    overlap_area: float
    offscreen_area: float
    minimum_size_loss: float
    def __init__(
        self,
        elements: Mapping[str, Element],
        size: tuple[float, float],
        overlap_area: float,
        offscreen_area: float,
        minimum_size_loss: float,
    ) -> None: ...
    def as_dict(self) -> dict[str, Any]: ...

class Layout:
    elements: Mapping[str, Element]
    size: tuple[float, float]
    inputs: Mapping[str, float]
    def __init__(
        self,
        elements: Iterable[Element] | Mapping[str, Element],
        size: tuple[Any, Any],
        inputs: Mapping[str, float] = ...,
    ) -> None: ...
    def snapshot(self, params: Mapping[str, Any] | None = ...) -> LayoutSnapshot: ...
    def overlap_loss(self, params: Mapping[str, Any] | None = ...) -> Tensor: ...
    def offscreen_loss(self, params: Mapping[str, Any] | None = ...) -> Tensor: ...
    def minimum_size_loss(self, params: Mapping[str, Any] | None = ...) -> Tensor: ...
    def containment_loss(self, params: Mapping[str, Any] | None = ...) -> Tensor: ...
    def content_overflow_loss(
        self, params: Mapping[str, Any] | None = ...
    ) -> Tensor: ...
    def violations(
        self, params: Mapping[str, Any] | None = ..., *, tolerance: float = ...
    ) -> dict[str, float]: ...
    def prevent_overlap(self, params: Mapping[str, Any] | None = ...) -> Tensor: ...
    def keep_inside_parent(self, params: Mapping[str, Any] | None = ...) -> Tensor: ...
    def is_valid(
        self, params: Mapping[str, Any] | None = ..., *, tolerance: float = ...
    ) -> bool: ...
    def assert_valid(
        self, params: Mapping[str, Any] | None = ..., *, tolerance: float = ...
    ) -> None: ...
    def size_loss(
        self,
        targets: Mapping[str, tuple[Any, Any]],
        params: Mapping[str, Any] | None = ...,
    ) -> Tensor: ...
    def alignment_loss(
        self,
        names: Iterable[str],
        axis: str = ...,
        target: Any = ...,
        params: Mapping[str, Any] | None = ...,
    ) -> Tensor: ...
    def gap_loss(
        self,
        names: Iterable[str],
        gap: Any,
        axis: str = ...,
        params: Mapping[str, Any] | None = ...,
    ) -> Tensor: ...

class LayoutTrainer:
    parameters: tuple[Parameter, ...]
    loss_fn: Callable[[], Tensor]
    optimizer: Any
    identity: str | None
    steps: int
    def __init__(
        self,
        parameters: Iterable[Parameter] | Mapping[str, Parameter],
        loss_fn: Callable[[], Tensor],
        optimizer: Any = ...,
        *,
        identity: Any = ...,
    ) -> None: ...
    def step(self) -> float: ...
    def evaluate(self) -> dict[str, float]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
    def save(self, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> None: ...
    def fit(
        self,
        steps: int,
        callback: Callable[[int, float], Any] | None = ...,
        *,
        validation: Callable[[], float | Mapping[str, Any]] | None = ...,
        patience: int | None = ...,
        min_delta: float = ...,
        restore_best: bool = ...,
        validation_interval: int = ...,
    ) -> list[float]: ...
