from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

class GPUImage:
    context: Any
    backend: Any
    texture: Any
    size: tuple[int, int]
    origin: tuple[int, int]
    canvas_size: tuple[int, int]
    format: str
    channels: int
    numpy_dtype: Any
    components: int
    dtype: str
    shape: tuple[int, ...]
    closed: bool
    generation: int
    def __init__(
        self,
        context: Any,
        texture: Any,
        *,
        size: tuple[int, int],
        format: str,
        origin: tuple[int, int] = ...,
        canvas_size: tuple[int, int] | None = ...,
    ) -> None: ...
    def write(self, data: Any) -> None: ...
    def read(self) -> NDArray[Any]: ...
    def read_bytes(self) -> bytes: ...
    def numpy(self) -> NDArray[Any]: ...
    def clone(self) -> GPUImage: ...
    def close(self) -> None: ...
    def __enter__(self) -> GPUImage: ...
    def __exit__(self, *_: Any) -> None: ...
