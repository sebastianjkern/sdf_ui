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
from pathlib import Path
from typing import Optional, Tuple

import wgpu


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
    items = tuple(value)
    if len(items) == 2:
        items = (*items, 1)
    if len(items) != 3 or any(int(item) <= 0 for item in items):
        raise ValueError(f"{name} must contain two or three positive values")
    return tuple(int(item) for item in items)


@dataclass(frozen=True)
class DispatchConfig:
    """Compute workgroup size and optional explicit dispatch count."""

    shader_local_size: Tuple[int, int, int] = (16, 16, 1)
    group_count: Optional[Tuple[int, int, int]] = None

    def __post_init__(self):
        object.__setattr__(self, "shader_local_size", _triple(self.shader_local_size, "shader_local_size"))
        if self.group_count is not None:
            object.__setattr__(self, "group_count", _triple(self.group_count, "group_count"))

    @classmethod
    def from_value(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("dispatch_config must be a DispatchConfig, dict, or None")

    def groups_for_size(self, size):
        if self.group_count is not None:
            return self.group_count
        width, height = size
        local_width, local_height, _ = self.shader_local_size
        return math.ceil(width / local_width), math.ceil(height / local_height), 1


class Context:
    """Own a WebGPU adapter, device, queue, and context-sized dispatch policy."""

    def __init__(self, size, dispatch_config=None, *, power_preference="high-performance"):
        self.size = tuple(int(value) for value in size)
        if len(self.size) != 2 or min(self.size) <= 0:
            raise ValueError("size must contain positive width and height")
        self._closed = False
        self.ctx = self
        self._training_graphs = weakref.WeakSet()
        self._image_generation = 0
        self.dispatch_config = DispatchConfig.from_value(dispatch_config)
        self.dispatch_groups = self.dispatch_config.groups_for_size(self.size)
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
        if self.adapter is None:
            raise RuntimeError("No WebGPU adapter is available")
        self.device = self.adapter.request_device_sync(label="sdf-ui device")
        self.queue = self.device.queue
        self._pipelines = {}
        self._pipeline_bindings = {}
        self._uniform_buffers = {}
        self._active_render_stats = None
        self.last_render_stats = None

    @property
    def device_info(self):
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
        """Allocate a buffer, initializing it when bytes-like data is supplied."""
        self._check_open()
        if isinstance(data_or_size, int):
            size = max(4, data_or_size)
            initial_data = None
        else:
            initial_data = bytes(data_or_size)
            size = max(4, (len(initial_data) + 3) & ~3)
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
        groups = self.dispatch_groups if group_count is None else _triple(group_count, "group_count")
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
    ):
        """Load a packaged WGSL compute shader and dispatch it by resource name."""
        path = self._shader_path(name, color_format)
        if not path.is_file():
            raise KeyError(f"Unknown WebGPU shader '{name}'")
        groups = self.run_compute(
            path.read_text(encoding="utf-8"),
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

    def run_node_shader(self, descriptor, output, inputs, uniforms, *, extra_outputs=()):
        """Execute a shader descriptor through its WGSL node kernel."""
        interface = descriptor.interface
        if interface is None:
            raise ValueError(f"Shader '{descriptor.name}' has no declared interface")
        if len(interface.inputs) != len(inputs):
            raise ValueError(f"Shader '{descriptor.name}' has invalid input count")
        resources = {interface.output.unit: output}
        if len(interface.extra_outputs) != len(extra_outputs):
            raise ValueError(f"Shader '{descriptor.name}' has invalid output count")
        for binding, image in zip(interface.extra_outputs, extra_outputs):
            resources[binding.unit] = image
        if isinstance(inputs, Mapping):
            if set(inputs) != {binding.name for binding in interface.inputs}:
                raise ValueError(f"Shader '{descriptor.name}' has invalid input names")
            images = (inputs[binding.name] for binding in interface.inputs)
        else:
            if len(interface.inputs) != len(inputs):
                raise ValueError(f"Shader '{descriptor.name}' has invalid input count")
            images = iter(inputs)
        for binding, image in zip(interface.inputs, images):
            resources[binding.unit] = image

        uniform_items = tuple(interface.uniforms)
        if uniform_items:
            payload = _pack_uniforms(uniform_items, uniforms)
            buffer_index = max(resources) + 1
            buffer = self._uniform_buffers.get((descriptor.name, buffer_index, len(payload)))
            if buffer is None:
                buffer = self.create_buffer(
                    payload,
                    usage=wgpu.BufferUsage.UNIFORM,
                    label=f"{descriptor.name} uniforms",
                )
                self._uniform_buffers[(descriptor.name, buffer_index, len(payload))] = buffer
            else:
                self.queue.write_buffer(buffer, 0, payload)
            resources[buffer_index] = buffer

        color_outputs = (output, *extra_outputs)
        color_format = (
            "rgba32f"
            if any(image.format == "rgba32float" for image in color_outputs)
            else "rgba8"
        )
        return self.run_shader(
            descriptor.name,
            resources,
            entry_point="compute_main",
            color_format=color_format,
        )

    def render(self, texture, params=None, cache=None):
        from ..core.texture import Renderer

        return Renderer(self, params=params, cache=cache).render(texture)

    def session(self, params=None, cache=None):
        from ..core.texture import Renderer

        return Renderer(self, params=params, cache=cache)

    def compile(self, outputs, params=None, *, size=None, inputs=None):
        """Compile named scene outputs for display or differentiable training."""
        from ..graph.runtime import RenderGraph

        return RenderGraph(self, outputs, params=params, size=size, inputs=inputs)

    def render_plan(self, plan, params=None):
        from .plan import execute_display

        return execute_display(plan, self, params=params)

    def create_graph_backend(self):
        from .backend import GPUShaderBackend

        return GPUShaderBackend(self)

    def training(self, outputs, params=None, *, size=None, inputs=None):
        from ..graph.session import TrainingSession

        return TrainingSession(self, outputs, params, size=size, inputs=inputs)

    def differentiate(self, texture, params=None):
        from ..learning.render import DifferentiableRenderer

        return DifferentiableRenderer(self.size, params=params).render(texture)

    def inspect_graph(self, outputs, params=None):
        from ..graph.inspection import inspect_graph

        return inspect_graph(outputs, params=params)

    @contextmanager
    def capture(self, scene, params=None, *, size=None, inputs=None):
        with self.compile(
            {"target": scene}, params=params, size=size, inputs=inputs
        ) as graph:
            with graph.snapshot("target") as image:
                yield image

    def allocate_buffer(self, size, *, label="graph buffer"):
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

    def create_image(self, size=None, *, format="r32float", label=""):
        """Allocate an owned image usable for storage, sampling, and copies."""
        self._check_open()
        width, height = self.size if size is None else size
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
            stats.record_texture_allocation(
                "sdf" if format == "r32float" else "color"
            )
        return GPUImage(self, texture, size=(width, height), format=format)

    def r32f(self, size=None):
        """Allocate a single-channel FP32 signed-distance image."""
        return self.create_image(size, format="r32float", label="sdf image")

    def rgba32f(self, size=None):
        """Allocate a four-channel FP32 working image for graph execution."""
        return self.create_image(size, format="rgba32float", label="rgba32f image")

    def rgba8(self, size=None):
        """Allocate a four-channel normalized image for display output."""
        return self.create_image(size, format="rgba8unorm", label="rgba8 image")

    def texture_from_image(self, path):
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

    def percent(self, alpha):
        return alpha / 100 * self.size[0]

    def percent_of_min(self, alpha):
        return alpha / 100 * min(self.size)

    def percent_x(self, alpha):
        return alpha / 100 * self.size[0]

    def percent_y(self, alpha):
        return alpha / 100 * self.size[1]

    def copy_image(self, image, *, label="image copy"):
        """Copy a GPU image without mapping or reading pixels on the CPU."""
        self._check_open()
        from .image import GPUImage

        if not isinstance(image, GPUImage) or image.context is not self:
            raise ValueError("Image must belong to this WebGPU context")
        image._check_open()
        result = self.create_image(image.size, format=image.format, label=label)
        encoder = self.device.create_command_encoder(label=label)
        encoder.copy_texture_to_texture(
            {"texture": image.texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"texture": result.texture, "mip_level": 0, "origin": (0, 0, 0)},
            (image.size[0], image.size[1], 1),
        )
        self.queue.submit([encoder.finish()])
        return result

    def close(self):
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

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
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
        array = re.fullmatch(r"(float|int|uint)\[(\d+)\]", uniform.glsl_type)
        if array is not None:
            alignment = 16
            count = int(array.group(2))
            fmt = {"float": "f", "int": "i", "uint": "I"}[array.group(1)]
            size = count * 16
        else:
            try:
                alignment, size, fmt = layouts[uniform.glsl_type]
            except KeyError as error:
                raise NotImplementedError(
                    f"Unsupported WGSL uniform type '{uniform.glsl_type}'"
                ) from error
            count = None
        offset = (offset + alignment - 1) // alignment * alignment
        fields.append((offset, size, fmt, uniform.name, uniform.glsl_type, count))
        offset += size
    total_size = (offset + 15) // 16 * 16
    payload = bytearray(total_size)
    for offset, _size, fmt, name, glsl_type, count in fields:
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
            continue
        if len(components) != count:
            raise ValueError(f"Uniform '{name}' requires {count} values")
        for index, component in enumerate(components):
            struct.pack_into("=" + fmt, payload, offset + index * 16, component)
    return bytes(payload)
