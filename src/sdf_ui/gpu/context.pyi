from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..core.texture import Renderer, TextureNode
from ..graph.runtime import FitResult, LossContext, RenderGraph
from ..graph.session import TrainingSession
from ..learning.autodiff import Tensor
from .image import GPUImage
from .resources import GPUBuffer

class DispatchConfig:
    shader_local_size: tuple[int, int, int]
    group_count: tuple[int, int, int] | None
    def __init__(
        self,
        shader_local_size: tuple[int, ...] = ...,
        group_count: tuple[int, ...] | None = ...,
    ) -> None: ...
    @classmethod
    def from_value(
        cls, value: DispatchConfig | dict[str, Any] | None
    ) -> DispatchConfig: ...
    def groups_for_size(self, size: tuple[int, int]) -> tuple[int, int, int]: ...

class Context:
    size: tuple[int, int]
    logical_size: tuple[int, int]
    device_scale: float
    progress: float
    dispatch_config: DispatchConfig
    dispatch_groups: tuple[int, int, int]
    device_info: dict[str, Any]
    def __init__(
        self,
        size: tuple[int, int],
        dispatch_config: DispatchConfig | dict[str, Any] | None = ...,
        *,
        power_preference: str = ...,
        device_scale: float = ...,
        enable_gpu_timing: bool = ...,
    ) -> None: ...
    def create_shader_module(self, code: str, *, label: str = ...) -> Any: ...
    def compute_pipeline(
        self, code: str, *, entry_point: str = ..., label: str = ...
    ) -> Any: ...
    def create_buffer(
        self,
        data_or_size: int | bytes | bytearray | memoryview,
        *,
        usage: Any,
        label: str = ...,
    ) -> Any: ...
    def write_buffer(self, buffer: Any, offset: int, data: bytes) -> None: ...
    def flush_commands(self) -> None: ...
    def bind_group(
        self,
        pipeline: Any,
        resources: Mapping[int, Any],
        *,
        group: int = ...,
        label: str = ...,
    ) -> Any: ...
    def dispatch(
        self,
        pipeline: Any,
        bind_groups: Iterable[Any],
        group_count: tuple[int, ...] | None = ...,
        *,
        label: str = ...,
        timestamp_name: str | None = ...,
    ) -> tuple[int, int, int]: ...
    def run_compute(
        self,
        code: str,
        resources: Mapping[int, Any],
        *,
        group_count: tuple[int, ...] | None = ...,
        entry_point: str = ...,
        label: str = ...,
        timestamp_name: str | None = ...,
    ) -> tuple[int, int, int]: ...
    def run_shader(
        self,
        name: str,
        resources: Mapping[int, Any],
        *,
        group_count: tuple[int, ...] | None = ...,
        entry_point: str = ...,
        color_format: str = ...,
        source_code: str | None = ...,
    ) -> tuple[int, int, int]: ...
    def supports_shader(self, name: str) -> bool: ...
    def run_node_shader(
        self,
        descriptor: Any,
        output: GPUImage,
        inputs: Any,
        uniforms: Any,
        *,
        extra_outputs: Iterable[GPUImage] = ...,
    ) -> tuple[int, int, int]: ...
    def render(
        self,
        texture: TextureNode,
        params: dict[str, Any] | None = ...,
        cache: dict[Any, Any] | None = ...,
        evaluation_rect: tuple[int, int, int, int] | None = ...,
        *,
        progress: float = ...,
    ) -> TextureNode: ...
    def session(
        self,
        params: dict[str, Any] | None = ...,
        cache: dict[Any, Any] | None = ...,
        evaluation_rect: tuple[int, int, int, int] | None = ...,
        *,
        progress: float = ...,
        dirty_regions: bool = ...,
    ) -> Renderer: ...
    def compile(
        self,
        outputs: Any,
        params: dict[str, Any] | None = ...,
        *,
        size: tuple[int, int] | None = ...,
        inputs: dict[str, Any] | None = ...,
    ) -> RenderGraph: ...
    def save_animation(
        self,
        scene: TextureNode,
        path: str | Path,
        *,
        frames: int = ...,
        params: dict[str, Any] | None = ...,
        fps: float = ...,
        dirty_regions: bool = ...,
    ) -> Path: ...
    def render_plan(
        self,
        plan: Any,
        params: dict[str, Any] | None = ...,
        evaluation_rect: tuple[int, int, int, int] | None = ...,
        *,
        progress: float = ...,
    ) -> Any: ...
    def training(
        self,
        outputs: Any,
        params: Any = ...,
        *,
        size: tuple[int, int] | None = ...,
        inputs: Any = ...,
    ) -> TrainingSession: ...
    def fit(
        self,
        outputs: Any,
        loss: Callable[[LossContext], Tensor],
        *,
        steps: int = ...,
        lr: float = ...,
        params: Mapping[str, Any] | None = ...,
        size: tuple[int, int] | None = ...,
        inputs: Mapping[str, Any] | None = ...,
        progress: float | None = ...,
    ) -> FitResult: ...
    def create_graph_backend(self) -> Any: ...
    def differentiate(
        self,
        texture: Any,
        params: dict[str, Any] | None = ...,
        *,
        progress: float = ...,
    ) -> Any: ...
    def inspect_graph(
        self, outputs: Any, params: dict[str, Any] | None = ...
    ) -> Any: ...
    def capture(
        self,
        scene: Any,
        params: dict[str, Any] | None = ...,
        *,
        size: tuple[int, int] | None = ...,
        inputs: dict[str, Any] | None = ...,
        progress: float | None = ...,
    ) -> AbstractContextManager[GPUImage]: ...
    def allocate_buffer(self, size: int, *, label: str = ...) -> GPUBuffer: ...
    def create_image(
        self,
        size: tuple[int, int] | None = ...,
        *,
        format: str = ...,
        label: str = ...,
        origin: tuple[int, int] = ...,
        canvas_size: tuple[int, int] | None = ...,
    ) -> GPUImage: ...
    def r32f(self, size: tuple[int, int] | None = ...) -> GPUImage: ...
    def rgba32f(self, size: tuple[int, int] | None = ...) -> GPUImage: ...
    def rgba8(self, size: tuple[int, int] | None = ...) -> GPUImage: ...
    def texture_from_image(self, path: str | Path) -> GPUImage: ...
    def percent(self, alpha: Any) -> Any: ...
    def percent_of_min(self, alpha: Any) -> Any: ...
    def percent_x(self, alpha: Any) -> Any: ...
    def percent_y(self, alpha: Any) -> Any: ...
    def copy_image(self, image: GPUImage, *, label: str = ...) -> GPUImage: ...
    def copy_image_region(
        self,
        source: GPUImage,
        destination: GPUImage,
        rect: tuple[int, int, int, int],
    ) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Context: ...
    def __exit__(self, *_: Any) -> None: ...
