from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Self

class PixelLoss:
    output: str
    target: Any
    kind: str
    weight: float
    delta: float
    name: str
    mask: Any
    normalization: str
    def __init__(
        self,
        output: str,
        target: Any,
        kind: str = ...,
        weight: float = ...,
        delta: float = ...,
        name: str = ...,
        mask: Any = ...,
        normalization: str = ...,
    ) -> None: ...

class GeometryLoss:
    callback: Callable[[], Any]
    weight: float
    name: str
    identity: Any
    def __init__(
        self,
        callback: Callable[[], Any],
        weight: float = ...,
        name: str = ...,
        identity: Any = ...,
    ) -> None: ...

class Objective:
    @property
    def pixel_losses(self) -> tuple[PixelLoss, ...]: ...
    @property
    def geometry_losses(self) -> tuple[GeometryLoss, ...]: ...
    @property
    def version(self) -> int: ...
    def __init__(self) -> None: ...
    def pixel(
        self,
        output: str,
        target: Any,
        *,
        kind: str = ...,
        weight: float = ...,
        delta: float = ...,
        name: str | None = ...,
        mask: Any = ...,
        normalization: str = ...,
    ) -> Self: ...
    def mse(
        self,
        output: str,
        target: Any,
        *,
        weight: float = ...,
        name: str | None = ...,
        mask: Any = ...,
        normalization: str = ...,
    ) -> Self: ...
    def mae(
        self,
        output: str,
        target: Any,
        *,
        weight: float = ...,
        name: str | None = ...,
        mask: Any = ...,
        normalization: str = ...,
    ) -> Self: ...
    def huber(
        self,
        output: str,
        target: Any,
        *,
        delta: float = ...,
        weight: float = ...,
        name: str | None = ...,
        mask: Any = ...,
        normalization: str = ...,
    ) -> Self: ...
    def geometry(
        self,
        callback: Callable[[], Any],
        *,
        weight: float = ...,
        name: str | None = ...,
        identity: Any = ...,
    ) -> Self: ...
    def set_target(self, name: str, target: Any) -> Self: ...
    def set_mask(self, name: str, mask: Any | None) -> Self: ...
    def set_weight(self, name: str, weight: float) -> Self: ...
    def remove(self, name: str) -> Self: ...
    def signature(
        self, *, portable: bool = ..., target_overrides: Mapping[Any, Any] | None = ...
    ) -> str: ...
