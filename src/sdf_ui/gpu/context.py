"""WebGPU device and compute execution primitives.

This module owns the explicit device, queue, shader, and texture operations
used by the WebGPU renderer. Graph and node execution are layered on top of
these primitives so backend details do not leak into the public scene API.
"""

from __future__ import annotations

import math
import re
import struct
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral, Number
from pathlib import Path
from typing import TYPE_CHECKING, Any

import wgpu

if TYPE_CHECKING:
    from ..core.texture import Renderer


def _wgsl_functions(code):
    functions = {}
    signature = re.compile(r"\bfn\s+(\w+)\s*\([^)]*\)[^{]*\{", re.DOTALL)
    for match in signature.finditer(code):
        depth = 1
        index = match.end()
        start = index
        while index < len(code) and depth:
            if code[index] == "{":
                depth += 1
            elif code[index] == "}":
                depth -= 1
            index += 1
        functions[match.group(1)] = code[start : index - 1]
    return functions


def _reachable_functions(functions, entry_point):
    reachable = set()
    pending = [entry_point]
    while pending:
        name = pending.pop()
        if name in reachable or name not in functions:
            continue
        reachable.add(name)
        body = functions[name]
        pending.extend(
            called
            for called in functions
            if called not in reachable
            and re.search(rf"\b{re.escape(called)}\s*\(", body)
        )
    return reachable


def _active_bindings(code, entry_point):
    """Find auto-layout bindings reachable from a WGSL entry point."""
    declarations = {}
    pattern = re.compile(
        r"@group\((\d+)\)\s*@binding\((\d+)\)\s*"
        r"var(?:<[^>]+>)?\s+(\w+)\b",
        re.MULTILINE,
    )
    for group, binding, name in pattern.findall(code):
        declarations.setdefault(int(group), {})[name] = int(binding)

    functions = _wgsl_functions(code)
    reachable = _reachable_functions(functions, entry_point)
    active_source = "\n".join(functions[name] for name in reachable)
    active = {}
    for group, names in declarations.items():
        active[group] = {
            binding
            for name, binding in names.items()
            if re.search(rf"\b{re.escape(name)}\b", active_source)
        }
    return active


def _triple(value, name):
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two or three positive values") from exc
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in items):
        raise ValueError(f"{name} must contain two or three positive integer values")
    items = tuple(int(item) for item in items)
    if len(items) == 2:
        items = (*items, 1)
    if len(items) != 3 or any(item <= 0 for item in items):
        raise ValueError(f"{name} must contain two or three positive values")
    return items


def _integer_pair(value, name, *, positive) -> tuple[int, int]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two integer values") from exc
    if len(items) != 2 or any(
        isinstance(item, bool) or not isinstance(item, Integral) for item in items
    ):
        raise ValueError(f"{name} must contain two integer values")
    items = (int(items[0]), int(items[1]))
    if positive and min(items) <= 0:
        raise ValueError(f"{name} must contain two positive integer values")
    if not positive and min(items) < 0:
        raise ValueError(f"{name} must contain two nonnegative integer values")
    return items


def _node_shader_images(descriptor, interface, output, inputs, extra_outputs):
    if len(interface.inputs) != len(inputs):
        raise ValueError(f"Shader '{descriptor.name}' has invalid input count")
    if len(interface.extra_outputs) != len(extra_outputs):
        raise ValueError(f"Shader '{descriptor.name}' has invalid output count")
    if isinstance(inputs, Mapping):
        expected = {binding.name for binding in interface.inputs}
        if set(inputs) != expected:
            raise ValueError(f"Shader '{descriptor.name}' has invalid input names")
        images = tuple(inputs[binding.name] for binding in interface.inputs)
    else:
        images = tuple(inputs)
    resources = {interface.output.unit: output}
    resources.update(
        (binding.unit, image)
        for binding, image in zip(interface.extra_outputs, extra_outputs)
    )
    resources.update(
        (binding.unit, image) for binding, image in zip(interface.inputs, images)
    )
    return resources, images


def _bind_node_uniforms(context, descriptor, resources, uniforms, values):
    payload = _pack_uniforms(uniforms, values)
    binding = max(resources) + 1
    key = (descriptor.name, binding, len(payload))
    buffer = context._uniform_buffers.get(key)
    if buffer is None:
        buffer = context.create_buffer(
            payload,
            usage=wgpu.BufferUsage.UNIFORM,
            label=f"{descriptor.name} uniforms",
        )
        context._uniform_buffers[key] = buffer
    else:
        context.queue.write_buffer(buffer, 0, payload)
    resources[binding] = buffer


@dataclass(frozen=True)
class DispatchConfig:
    """Compute workgroup size and optional explicit dispatch count."""

    shader_local_size: tuple[int, int, int] = (16, 16, 1)
    group_count: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shader_local_size",
            _triple(self.shader_local_size, "shader_local_size"),
        )
        if self.group_count is not None:
            object.__setattr__(
                self, "group_count", _triple(self.group_count, "group_count")
            )

    @classmethod
    def from_value(
        cls, value: DispatchConfig | dict[str, Any] | None
    ) -> DispatchConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("dispatch_config must be a DispatchConfig, dict, or None")

    def groups_for_size(self, size: tuple[int, int]) -> tuple[int, int, int]:
        if self.group_count is not None:
            return self.group_count
        width, height = size
        local_width, local_height, _ = self.shader_local_size
        return math.ceil(width / local_width), math.ceil(height / local_height), 1


class Context:
    """Own a WebGPU adapter, device, queue, and context-sized dispatch policy."""

    def __init__(
        self,
        size: tuple[int, int],
        dispatch_config: DispatchConfig | dict[str, Any] | None = None,
        *,
        power_preference: str = "high-performance",
    ) -> None:
        try:
            values = tuple(size)
        except TypeError as exc:
            raise ValueError(
                "size must be a pair of positive integer width and height values"
            ) from exc
        if len(values) != 2 or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in values
        ):
            raise ValueError(
                "size must be a pair of positive integer width and height values"
            )
        self.size = (int(values[0]), int(values[1]))
        if min(self.size) <= 0:
            raise ValueError(
                "size must be a pair of positive integer width and height values"
            )
        self._closed = False
        self.ctx = self
        self._training_graphs: weakref.WeakSet[Any] = weakref.WeakSet()
        self._image_generation = 0
        self.dispatch_config = DispatchConfig.from_value(dispatch_config)
        self.dispatch_groups = self.dispatch_config.groups_for_size(self.size)
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
        if self.adapter is None:
            raise RuntimeError("No WebGPU adapter is available")
        self.device = self.adapter.request_device_sync(label="sdf-ui device")
        self.queue = self.device.queue
        self._pipelines: dict[Any, Any] = {}
        self._pipeline_bindings: dict[int, Any] = {}
        self._uniform_buffers: dict[Any, Any] = {}
        self._active_evaluation_rect = None
        self._active_render_stats = None
        self.last_render_stats = None

    @property
    def device_info(self) -> dict[str, Any]:
        return dict(self.adapter.info)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("WebGPU context is closed")

    def create_shader_module(self, code, *, label=""):
        self._check_open()
        return self.device.create_shader_module(label=label, code=code)

    def compute_pipeline(self, code, *, entry_point="main", label=""):
        """Return a cached auto-layout compute pipeline for WGSL source."""
        self._check_open()
        key = (code, entry_point)
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            module = self.device.create_shader_module(label=label, code=code)
            pipeline = self.device.create_compute_pipeline(
                label=label,
                layout="auto",
                compute={"module": module, "entry_point": entry_point},
            )
            self._pipelines[key] = pipeline
            self._pipeline_bindings[id(pipeline)] = _active_bindings(code, entry_point)
        return pipeline

    def create_buffer(self, data_or_size, *, usage, label=""):
        """Allocate a buffer, zero-padding bytes-like initial data to four bytes."""
        self._check_open()
        if isinstance(data_or_size, Integral) and not isinstance(data_or_size, bool):
            if data_or_size < 0:
                raise ValueError("buffer size must be a nonnegative integer")
            size = max(4, int(data_or_size))
            initial_data = None
        else:
            if isinstance(data_or_size, Number):
                raise ValueError("buffer size must be a nonnegative integer")
            if not isinstance(data_or_size, (bytes, bytearray, memoryview)):
                raise TypeError(
                    "buffer data must be a nonnegative integer size or bytes-like"
                )
            initial_data = bytes(data_or_size)
            size = max(4, (len(initial_data) + 3) & ~3)
            initial_data = initial_data.ljust(size, b"\0")
        buffer_usage = usage
        if initial_data is not None:
            buffer_usage |= wgpu.BufferUsage.COPY_DST
        buffer = self.device.create_buffer(
            label=label,
            size=size,
            usage=buffer_usage,
        )
        if initial_data:
            self.queue.write_buffer(buffer, 0, initial_data)
        return buffer

    def bind_group(self, pipeline, resources, *, group=0, label=""):
        """Create a bind group from binding-indexed buffers or image views."""
        self._check_open()
        from .image import GPUImage
        from .resources import GPUBuffer

        entries = []
        active = self._pipeline_bindings.get(id(pipeline), {}).get(group)
        for binding, resource in sorted(resources.items()):
            if active is not None and binding not in active:
                continue
            if isinstance(resource, GPUImage):
                resource._check_open()
                resource = resource.texture.create_view()
            elif isinstance(resource, GPUBuffer):
                resource._check_open()
                resource = resource.buffer
            entries.append({"binding": binding, "resource": resource})
        return self.device.create_bind_group(
            label=label,
            layout=pipeline.get_bind_group_layout(group),
            entries=entries,
        )

    def dispatch(self, pipeline, bind_groups, group_count=None, *, label=""):
        """Encode and submit one compute pass with its ordered bind groups."""
        self._check_open()
        groups = (
            self.dispatch_groups
            if group_count is None
            else _triple(group_count, "group_count")
        )
        encoder = self.device.create_command_encoder(label=label)
        compute = encoder.begin_compute_pass(label=label)
        compute.set_pipeline(pipeline)
        for index, bind_group in enumerate(bind_groups):
            compute.set_bind_group(index, bind_group)
        compute.dispatch_workgroups(*groups)
        compute.end()
        self.queue.submit([encoder.finish()])
        return groups

    def run_compute(
        self,
        code,
        resources,
        *,
        group_count=None,
        entry_point="main",
        label="",
    ):
        """Compile or reuse a WGSL pipeline and dispatch its resource bindings."""
        pipeline = self.compute_pipeline(code, entry_point=entry_point, label=label)
        bind_group = self.bind_group(pipeline, resources, label=label)
        return self.dispatch(
            pipeline,
            (bind_group,),
            group_count=group_count,
            label=label,
        )

    def _shader_path(self, name, color_format="rgba8"):
        if not name or Path(name).name != name:
            raise ValueError("Shader name must be a simple file name")
        shader_dir = Path(__file__).with_name("shaders")
        if color_format == "rgba32f":
            variant = shader_dir / f"{name}_rgba32f.wgsl"
            if variant.is_file():
                return variant
        return shader_dir / f"{name}.wgsl"

    def run_shader(
        self,
        name,
        resources,
        *,
        group_count=None,
        entry_point="main",
        color_format="rgba8",
        source_code=None,
    ):
        """Load a packaged WGSL compute shader and dispatch it by resource name."""
        path = self._shader_path(name, color_format)
        if not path.is_file():
            raise KeyError(f"Unknown WebGPU shader '{name}'")
        code = path.read_text(encoding="utf-8") if source_code is None else source_code
        groups = self.run_compute(
            code,
            resources,
            group_count=group_count,
            entry_point=entry_point,
            label=name,
        )
        stats = self._active_render_stats
        if stats is not None:
            stats.record_shader_dispatch(name, groups)
        return groups

    def supports_shader(self, name):
        return (Path(__file__).with_name("shaders") / f"{name}.wgsl").is_file()

    def run_node_shader(
        self, descriptor, output, inputs, uniforms, *, extra_outputs=()
    ):
        """Execute a shader descriptor through its WGSL node kernel."""
        interface = descriptor.interface
        if interface is None:
            raise ValueError(f"Shader '{descriptor.name}' has no declared interface")
        resources, images = _node_shader_images(
            descriptor, interface, output, inputs, extra_outputs
        )
        uniform_items = tuple(interface.uniforms)
        if uniform_items:
            _bind_node_uniforms(self, descriptor, resources, uniform_items, uniforms)

        color_outputs = (output, *extra_outputs)
        color_format = (
            "rgba32f"
            if any(image.format == "rgba32float" for image in color_outputs)
            else "rgba8"
        )
        source_code, group_count = self._region_shader(
            descriptor, interface, resources, images, color_format
        )
        return self.run_shader(
            descriptor.name,
            resources,
            entry_point="compute_main",
            color_format=color_format,
            group_count=group_count,
            source_code=source_code,
        )

    def _region_shader(self, descriptor, interface, resources, images, color_format):
        rect = self._active_evaluation_rect
        if rect is None:
            return None, None
        contract = descriptor.differentiation
        if contract is None or contract.sampling != "pointwise":
            sampling = None if contract is None else contract.sampling
            raise NotImplementedError(
                f"Evaluation rectangles require pointwise shader sampling; "
                f"'{descriptor.name}' uses {sampling}"
            )
        source_code = self._shader_path(descriptor.name, color_format).read_text(
            encoding="utf-8"
        )
        bindings = re.findall(r"@group\(0\)\s*@binding\((\d+)\)", source_code)
        region_binding = max(map(int, bindings), default=-1) + 1
        from .regions import pack_rect, rect_groups, specialize_wgsl_for_rect

        source_code = specialize_wgsl_for_rect(source_code, interface, region_binding)
        payload = pack_rect(self.size, rect, images)
        key = (descriptor.name, "evaluation region", region_binding, len(payload))
        buffer = self._uniform_buffers.get(key)
        if buffer is None:
            buffer = self.create_buffer(
                payload,
                usage=wgpu.BufferUsage.UNIFORM,
                label=f"{descriptor.name} evaluation region",
            )
            self._uniform_buffers[key] = buffer
        else:
            self.queue.write_buffer(buffer, 0, payload)
        resources[region_binding] = buffer
        return source_code, rect_groups(source_code, rect)

    def render(
        self,
        texture: Any,
        params: dict[str, Any] | None = None,
        cache: dict[Any, Any] | None = None,
        evaluation_rect: tuple[int, int, int, int] | None = None,
    ) -> Any:
        """Render a node, optionally returning a cropped pixel rectangle.

        ``evaluation_rect`` is ``(x, y, width, height)`` in canvas pixels. It
        applies to every shader in the render tree and supports pointwise
        shaders; neighborhood-sampling and host-rendered operations reject it.
        """
        from ..core.texture import Renderer  # type: ignore[attr-defined]

        return Renderer(
            self,
            params=params,
            cache=cache,
            evaluation_rect=evaluation_rect,
        ).render(texture)

    def session(
        self,
        params: dict[str, Any] | None = None,
        cache: dict[Any, Any] | None = None,
        evaluation_rect: tuple[int, int, int, int] | None = None,
    ) -> Renderer:
        from ..core.texture import Renderer

        return Renderer(
            self,
            params=params,
            cache=cache,
            evaluation_rect=evaluation_rect,
        )

    def compile(
        self,
        outputs: Any,
        params: dict[str, Any] | None = None,
        *,
        size: tuple[int, int] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> Any:
        """Compile named scene outputs for display or differentiable training."""
        from ..graph.runtime import RenderGraph

        return RenderGraph(self, outputs, params=params, size=size, inputs=inputs)

    def render_plan(
        self,
        plan: Any,
        params: dict[str, Any] | None = None,
        evaluation_rect: tuple[int, int, int, int] | None = None,
    ) -> Any:
        from .plan import execute_display

        with self._evaluation_scope(evaluation_rect):
            return execute_display(plan, self, params=params)

    @contextmanager
    def _evaluation_scope(self, evaluation_rect):
        from .regions import normalize_rect

        self._check_open()
        previous = self._active_evaluation_rect
        self._active_evaluation_rect = normalize_rect(evaluation_rect, self.size)
        try:
            yield self._active_evaluation_rect
        finally:
            self._active_evaluation_rect = previous

    def create_graph_backend(self):
        from .backend import GPUShaderBackend

        return GPUShaderBackend(self)

    def training(
        self,
        outputs: Any,
        params: Any = None,
        *,
        size: tuple[int, int] | None = None,
        inputs: Any = None,
    ) -> Any:
        from ..graph.session import TrainingSession

        return TrainingSession(self, outputs, params, size=size, inputs=inputs)

    def differentiate(self, texture: Any, params: dict[str, Any] | None = None) -> Any:
        from ..learning.render import DifferentiableRenderer

        return DifferentiableRenderer(self.size, params=params).render(texture)

    def inspect_graph(self, outputs: Any, params: dict[str, Any] | None = None) -> Any:
        from ..graph.inspection import inspect_graph

        return inspect_graph(outputs, params=params)

    @contextmanager
    def capture(
        self,
        scene: Any,
        params: dict[str, Any] | None = None,
        *,
        size: tuple[int, int] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> Any:
        with self.compile(
            {"target": scene}, params=params, size=size, inputs=inputs
        ) as graph:
            with graph.snapshot("target") as image:
                yield image

    def allocate_buffer(self, size: int, *, label: str = "graph buffer") -> Any:
        self._check_open()
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 0:
            raise ValueError("buffer size must be a nonnegative integer")
        from .resources import GPUBuffer

        buffer = self.device.create_buffer(
            label=label,
            size=max(4, (int(size) + 3) & ~3),
            usage=(
                wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.COPY_DST
            ),
        )
        return GPUBuffer(self, buffer)

    def create_image(
        self,
        size: tuple[int, int] | None = None,
        *,
        format: str = "r32float",
        label: str = "",
        origin: tuple[int, int] = (0, 0),
        canvas_size: tuple[int, int] | None = None,
    ) -> Any:
        """Allocate an owned image usable for storage, sampling, and copies."""
        self._check_open()
        size_value = (
            None if size is None else _integer_pair(size, "size", positive=True)
        )
        origin_value = _integer_pair(origin, "origin", positive=False)
        canvas_size_value = _integer_pair(
            canvas_size
            if canvas_size is not None
            else (self.size if size_value is None else size_value),
            "canvas_size",
            positive=True,
        )
        if self._active_evaluation_rect is not None and (
            size_value is None or size_value == self.size
        ):
            x, y, width, height = self._active_evaluation_rect
            origin_value = (x, y)
            size_value = (width, height)
            canvas_size_value = self.size
        width, height = self.size if size_value is None else size_value
        if (
            origin_value[0] + width > canvas_size_value[0]
            or origin_value[1] + height > canvas_size_value[1]
        ):
            raise ValueError("image extent and origin must fit inside canvas_size")
        texture = self.device.create_texture(
            label=label,
            size=(int(width), int(height), 1),
            dimension="2d",
            format=format,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.COPY_DST
            ),
        )
        from .image import GPUImage

        stats = self._active_render_stats
        if stats is not None:
            stats.record_texture_allocation("sdf" if format == "r32float" else "color")
        return GPUImage(
            self,
            texture,
            size=(width, height),
            format=format,
            origin=origin_value,
            canvas_size=canvas_size_value,
        )

    def r32f(self, size: tuple[int, int] | None = None) -> Any:
        """Allocate a single-channel FP32 signed-distance image."""
        return self.create_image(size, format="r32float", label="sdf image")

    def rgba32f(self, size: tuple[int, int] | None = None) -> Any:
        """Allocate a four-channel FP32 working image for graph execution."""
        return self.create_image(size, format="rgba32float", label="rgba32f image")

    def rgba8(self, size: tuple[int, int] | None = None) -> Any:
        """Allocate a four-channel normalized image for display output."""
        return self.create_image(size, format="rgba8unorm", label="rgba8 image")

    def texture_from_image(self, path: str | Path) -> Any:
        """Load an image file as a vertically flipped RGBA8 GPU image."""
        import cv2
        import numpy as np

        image_path = Path(path)
        pixels = cv2.imread(str(image_path))
        if pixels is None:
            raise FileNotFoundError(f"Unable to load image from {image_path}")
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGBA)
        pixels = np.flip(pixels, axis=0).copy(order="C")
        image = self.rgba8((pixels.shape[1], pixels.shape[0]))
        image.write(pixels)
        return image

    def percent(self, alpha: Any) -> Any:
        """Return ``alpha`` percent of the canvas width."""
        return alpha / 100 * self.size[0]

    def percent_of_min(self, alpha: Any) -> Any:
        """Return ``alpha`` percent of the smaller canvas dimension."""
        return alpha / 100 * min(self.size)

    def percent_x(self, alpha: Any) -> Any:
        """Return ``alpha`` percent of the canvas width."""
        return alpha / 100 * self.size[0]

    def percent_y(self, alpha: Any) -> Any:
        """Return ``alpha`` percent of the canvas height."""
        return alpha / 100 * self.size[1]

    def copy_image(self, image: Any, *, label: str = "image copy") -> Any:
        """Copy a GPU image without mapping or reading pixels on the CPU."""
        self._check_open()
        from .image import GPUImage

        if not isinstance(image, GPUImage) or image.context is not self:
            raise ValueError("Image must belong to this WebGPU context")
        image._check_open()
        result = self.create_image(
            image.size,
            format=image.format,
            label=label,
            origin=image.origin,
            canvas_size=image.canvas_size,
        )
        encoder = self.device.create_command_encoder(label=label)
        encoder.copy_texture_to_texture(
            {"texture": image.texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"texture": result.texture, "mip_level": 0, "origin": (0, 0, 0)},
            (image.size[0], image.size[1], 1),
        )
        self.queue.submit([encoder.finish()])
        return result

    def close(self) -> None:
        if self._closed:
            return
        for graph in tuple(self._training_graphs):
            graph.close()
        for buffer in self._uniform_buffers.values():
            buffer.destroy()
        self._pipelines.clear()
        self._uniform_buffers.clear()
        self._closed = True
        self.device.destroy()

    def __enter__(self) -> Context:
        self._check_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _pack_uniforms(uniforms, values):
    """Pack scalar and vector shader uniforms using WGSL struct alignment."""
    layouts = {
        "float": (4, 4, "f"),
        "int": (4, 4, "i"),
        "uint": (4, 4, "I"),
        "bool": (4, 4, "I"),
        "vec2": (8, 8, "2f"),
        "vec3": (16, 12, "3f"),
        "vec4": (16, 16, "4f"),
        "ivec2": (8, 8, "2i"),
        "ivec3": (16, 12, "3i"),
        "ivec4": (16, 16, "4i"),
        "uvec2": (8, 8, "2I"),
        "uvec3": (16, 12, "3I"),
        "uvec4": (16, 16, "4I"),
    }
    fields = []
    offset = 0
    for uniform in uniforms:
        alignment, size, fmt, count = _uniform_layout(uniform, layouts)
        offset = (offset + alignment - 1) // alignment * alignment
        fields.append((offset, size, fmt, uniform.name, uniform.glsl_type, count))
        offset += size
    total_size = (offset + 15) // 16 * 16
    payload = bytearray(total_size)
    for offset, _size, fmt, name, glsl_type, count in fields:
        _pack_uniform_field(payload, values, offset, fmt, name, glsl_type, count)
    return bytes(payload)


def _uniform_layout(uniform, layouts):
    array = re.fullmatch(r"(float|int|uint)\[(\d+)\]", uniform.glsl_type)
    if array is not None:
        count = int(array.group(2))
        fmt = {"float": "f", "int": "i", "uint": "I"}[array.group(1)]
        return 16, count * 16, fmt, count
    try:
        alignment, size, fmt = layouts[uniform.glsl_type]
    except KeyError as error:
        raise NotImplementedError(
            f"Unsupported WGSL uniform type '{uniform.glsl_type}'"
        ) from error
    return alignment, size, fmt, None


def _pack_uniform_field(payload, values, offset, fmt, name, glsl_type, count):
    value = values.get(name, 0)
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "reshape"):
        value = value.reshape(-1).tolist()
    components = value if isinstance(value, (tuple, list)) else (value,)
    if name not in values:
        components = (0,) * (count or struct.calcsize(fmt) // 4)
    if glsl_type == "bool":
        components = tuple(int(bool(component)) for component in components)
    if count is None:
        struct.pack_into("=" + fmt, payload, offset, *components)
        return
    if len(components) != count:
        raise ValueError(f"Uniform '{name}' requires {count} values")
    for index, component in enumerate(components):
        struct.pack_into("=" + fmt, payload, offset + index * 16, component)
