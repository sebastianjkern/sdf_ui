"""Render-graph execution over WebGPU images, buffers, and WGSL kernels."""

from __future__ import annotations

import struct
import re
from hashlib import sha256
from pathlib import Path

import numpy as np
import wgpu

from ..core.image import shape_size


class GPUShaderBackend:
    """Execute differentiable shader descriptors on a WebGPU context."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.ctx = canvas
        self.descriptors = {item.name: item for item in _shader_registry().shader_files()}
        self._closed = False
        self._scratch = {}
        self._uniform_buffers = {}
        self.status = self.create_buffer(4)
        self.stats = {
            "forward_dispatches": 0,
            "backward_dispatches": 0,
            "reduction_dispatches": 0,
            "loss_dispatches": 0,
            "image_readbacks": 0,
            "image_uploads": 0,
            "program_compilations": 0,
        }
        canvas._gpu_stats = self.stats
        self.status.clear()

    def _check(self):
        if self._closed:
            raise RuntimeError("WebGPU graph backend is closed")
        self.canvas._check_open()

    @property
    def device_info(self):
        return self.canvas.device_info

    def create_buffer(self, size):
        self._check()
        return self.canvas.allocate_buffer(size)

    def owns_context(self, context):
        return not getattr(context, "_closed", False) and context is self.canvas

    def create_optimizer(self, parameters, lr):
        from .optim import GPUAdam

        return GPUAdam(self.canvas, parameters, lr=lr)

    def scratch_buffer(self, name, size):
        self._check()
        size = max(4, int(size))
        buffer = self._scratch.get(name)
        if buffer is None or buffer.size < size:
            replacement = self.create_buffer(size)
            if buffer is not None:
                buffer.close()
            self._scratch[name] = replacement
            buffer = replacement
        return buffer

    def _uniform_buffer(self, key, payload):
        buffer = self._uniform_buffers.get(key)
        if buffer is None or buffer.size < len(payload):
            if buffer is not None:
                buffer.destroy()
            buffer = self.canvas.create_buffer(
                payload,
                usage=wgpu.BufferUsage.UNIFORM,
                label=key,
            )
            self._uniform_buffers[key] = buffer
        else:
            self.canvas.queue.write_buffer(buffer, 0, payload)
        return buffer

    def contract(self, name):
        self._check()
        try:
            descriptor = self.descriptors[name]
        except KeyError as error:
            raise ValueError(f"Unknown WebGPU shader '{name}'") from error
        contract = descriptor.differentiation
        if contract is None:
            raise ValueError(f"Shader '{name}' has no differentiation contract")
        return contract

    def defaults(self, name):
        return getattr(self.contract(name), "defaults", None) or {}

    def uniforms(self, name, params):
        contract = self.contract(name)
        params = {} if params is None else dict(params)
        arguments = {argument.name: argument for argument in contract.parameters}
        unknown = set(params) - set(arguments)
        if unknown:
            raise ValueError(f"Unknown uniforms for '{name}': {sorted(unknown)}")
        defaults = self.defaults(name)
        result = {}
        for argument in contract.parameters:
            value = params.get(
                argument.name,
                defaults.get(argument.name, np.zeros(argument.shape)),
            )
            if hasattr(value, "numpy"):
                value = value.numpy()
            value = np.asarray(value, dtype=np.float32)
            if value.shape != argument.shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"Uniform '{argument.name}' must have finite shape {argument.shape}"
                )
            result[argument.name] = value.copy()
        return result

    def shader_fingerprints(self, name):
        self.contract(name)
        forward = self._shader_path(name, "rgba32f")
        backward = self._shader_path(name, backward=True)
        return (
            sha256(backward.read_bytes()).hexdigest(),
            sha256(forward.read_bytes()).hexdigest(),
        )

    def derivative_code(self, name):
        self.contract(name)
        return self._shader_path(name, backward=True).read_text(encoding="utf-8")

    def _shader_path(self, name, color_format="rgba8", backward=False):
        shader_dir = Path(__file__).with_name("shaders")
        if backward:
            path = shader_dir / f"{name}_backward.wgsl"
        elif color_format == "rgba32f":
            path = shader_dir / f"{name}_rgba32f.wgsl"
            if not path.is_file():
                path = shader_dir / f"{name}.wgsl"
        else:
            path = shader_dir / f"{name}.wgsl"
        if not path.is_file():
            raise NotImplementedError(f"WGSL shader '{path.name}' is not available")
        return path

    def upload(self, pixels):
        self._check()
        pixels = np.asarray(pixels, dtype=np.float32)
        if pixels.ndim == 2:
            image = self.canvas.r32f((pixels.shape[1], pixels.shape[0]))
        elif pixels.ndim == 3 and pixels.shape[2] == 4:
            image = self.canvas.rgba32f((pixels.shape[1], pixels.shape[0]))
        else:
            raise ValueError("Images must have shape (height,width) or (height,width,4)")
        if not pixels.size or not np.all(np.isfinite(pixels)):
            image.close()
            raise ValueError("Image dimensions must be positive and values finite")
        image.write(pixels)
        self.stats["image_uploads"] += 1
        return image

    def allocate(self, size, channels):
        self._check()
        size = tuple(size)
        if channels not in (1, 4) or len(size) != 2 or any(int(v) != v or v <= 0 for v in size):
            raise ValueError("FP32 images require positive dimensions and 1 or 4 channels")
        return self.canvas.r32f(size) if channels == 1 else self.canvas.rgba32f(size)

    def copy_image(self, image):
        self._check()
        if image.closed or image.context is not self.canvas:
            raise ValueError("Cannot copy a closed image or one from another context")
        return self.canvas.copy_image(image)

    def import_texture(self, texture, channels):
        self._check()
        if getattr(texture, "context", None) is not self.canvas:
            raise ValueError("Graph textures must belong to this WebGPU context")
        if texture.closed or texture.channels != channels:
            raise ValueError("Graph texture has an incompatible format")
        if texture.format in {"r32float", "rgba32float"}:
            return self.copy_image(texture)
        if channels == 4 and texture.format == "rgba8unorm":
            output = self.canvas.rgba32f(texture.size)
            try:
                self.canvas.run_shader(
                    "import_color",
                    {0: output, 1: texture},
                )
            except BaseException:
                output.close()
                raise
            return output
        raise NotImplementedError("Importing normalized textures into FP32 graph storage is pending")

    def display_image(self, image, mode=None):
        self._check()
        if image.closed or image.context is not self.canvas:
            raise ValueError("Cannot display a closed image or one from another context")
        if image.channels == 1:
            from ..core.sdf import SDFTexture

            texture = self.canvas.copy_image(image)
            return SDFTexture(tex=texture, context=self.canvas)
        from ..core.color import ColorTexture

        texture = self.canvas.rgba8(image.size)
        try:
            self.canvas.run_shader("display_color", {0: texture, 1: image})
        except BaseException:
            texture.close()
            raise
        return ColorTexture(tex=texture, context=self.canvas, mode=mode or "LAB")

    def validate_images(self, images):
        self._check()
        for image in images:
            if image.closed or image.context is not self.canvas:
                raise ValueError("Graph image is closed or belongs to another context")
            if image.format not in {"r32float", "rgba32float"}:
                raise ValueError("Graph execution requires FP32 images")
            name = "validate_sdf" if image.channels == 1 else "validate_color"
            groups = ((image.size[0] * image.size[1] + 63) // 64, 1, 1)
            self.canvas.run_shader(
                name,
                {0: image, 17: self.status},
                group_count=groups,
            )

    def forward(self, name, inputs=None, params=None, size=None):
        contract = self.contract(name)
        inputs = {} if inputs is None else dict(inputs)
        expected = {argument.name for argument in contract.inputs}
        if set(inputs) != expected:
            raise ValueError(f"Shader '{name}' input names differ from its contract")
        if size is None:
            size = next(iter(inputs.values())).size if inputs else self.canvas.size
        size = tuple(size)
        if len(size) != 2 or any(int(value) != value or value <= 0 for value in size):
            raise ValueError("GPU image size must be two positive integers")
        for argument in contract.inputs:
            image = inputs[argument.name]
            if image.closed or image.context is not self.canvas or image.size != size:
                raise ValueError(f"Incompatible GPU input image '{argument.name}'")
            if image.channels != shape_size(argument.shape):
                raise ValueError(f"Incompatible GPU input image '{argument.name}'")

        outputs = tuple(
            self.allocate(size, shape_size(argument.shape))
            for argument in contract.outputs
        )
        try:
            descriptor = self.descriptors[name]
            self.canvas.run_node_shader(
                descriptor,
                outputs[0],
                inputs,
                self.uniforms(name, params),
                extra_outputs=outputs[1:],
            )
        except BaseException:
            for image in outputs:
                image.close()
            raise
        self.stats["forward_dispatches"] += 1
        return outputs

    def clear_status(self):
        self.status.clear()

    def check_status(self):
        code = int(np.frombuffer(self.status.read(4), dtype=np.uint32)[0])
        if code:
            raise FloatingPointError(f"WebGPU differentiation failed (status={code})")

    def mask_weight_sum(self, image):
        from .loss import mask_weight_sum

        return mask_weight_sum(self, image)

    def seed_pixel_loss(self, *args):
        from .loss import seed_pixel_loss

        return seed_pixel_loss(self, *args)

    def backward_into(
        self,
        name,
        inputs,
        params,
        adjoints,
        input_offsets,
        output_offsets,
        uniform_gradients,
        result_offset=0,
        *,
        size=None,
    ):
        contract = self.contract(name)
        uniforms = self.uniforms(name, params)
        parameter_values = (
            np.concatenate(
                [uniforms[arg.name].reshape(-1) for arg in contract.parameters]
            )
            if contract.parameters
            else np.zeros(1, dtype=np.float32)
        )
        components = sum(shape_size(arg.shape) for arg in contract.parameters)
        if len(input_offsets) != len(contract.inputs) or len(output_offsets) != len(
            contract.outputs
        ):
            raise ValueError("GPU adjoint offsets differ from shader port counts")
        if size is None:
            size = next(iter(inputs.values())).size if inputs else self.canvas.size
        groups = (size[0] * size[1] + 63) // 64
        parameter_buffer = self.scratch_buffer(
            "backward_parameters", parameter_values.nbytes
        )
        parameter_buffer.write(parameter_values.tobytes())
        partials = self.scratch_buffer(
            "backward_partials", groups * max(components, 1) * 4
        )
        resources = {
            index: inputs[arg.name]
            for index, arg in enumerate(contract.inputs)
        }
        source = self.derivative_code(name)
        backward_uniforms = self._uniform_buffer(
            f"{name} backward uniforms",
            _pack_backward_uniforms(source, size, input_offsets, output_offsets),
        )
        resources.update(
            {
                _uniform_binding(source): backward_uniforms,
                16: parameter_buffer,
                17: adjoints,
                18: partials,
                20: self.status,
            }
        )
        pipeline = self.canvas.compute_pipeline(
            source, entry_point="compute_main", label=f"{name} backward"
        )
        bind_group = self.canvas.bind_group(pipeline, resources, label=f"{name} backward")
        self.canvas.dispatch(
            pipeline,
            (bind_group,),
            group_count=(groups, 1, 1),
            label=f"{name} backward",
        )
        self.stats["backward_dispatches"] += 1

        if components:
            config = np.asarray(
                [groups, components, result_offset, 0], dtype=np.uint32
            ).tobytes()
            config_buffer = self._uniform_buffer(
                "uniform gradient reduction config", config
            )
            self.canvas.run_shader(
                "reduce_uniform",
                {
                    0: partials,
                    1: uniform_gradients,
                    2: config_buffer,
                    3: self.status,
                },
                group_count=((components + 63) // 64, 1, 1),
            )
            self.stats["reduction_dispatches"] += 1

    def close(self):
        if self._closed:
            return
        self.status.close()
        for buffer in self._uniform_buffers.values():
            buffer.destroy()
        self._uniform_buffers.clear()
        for buffer in self._scratch.values():
            buffer.close()
        self._scratch.clear()
        self._closed = True


def _shader_registry():
    from .nodes.registry import registry

    return registry


def _pack_backward_uniforms(source, size, input_offsets, output_offsets):
    """Pack the generated WGSL derivative entry-point uniform struct."""
    match = re.search(r"struct Uniforms\s*\{([^}]*)\}", source, re.DOTALL)
    if match is None:
        raise ValueError("Backward WGSL kernel has no Uniforms struct")
    field_names = re.findall(
        r"\b(size|inputOffsets|outputOffsets|uniformBase)\s*:", match.group(1)
    )
    payload = bytearray(256)
    offsets = {}
    cursor = 0
    for name in field_names:
        if name == "size":
            cursor = (cursor + 7) // 8 * 8
            offsets[name] = cursor
            cursor += 8
        elif name in {"inputOffsets", "outputOffsets"}:
            cursor = (cursor + 15) // 16 * 16
            offsets[name] = cursor
            count = len(input_offsets if name == "inputOffsets" else output_offsets)
            cursor += max(count, 1) * 16
        else:
            cursor = (cursor + 3) // 4 * 4
            offsets[name] = cursor
            cursor += 4
    struct.pack_into("<2i", payload, offsets["size"], *size)
    for name, values in (
        ("inputOffsets", input_offsets),
        ("outputOffsets", output_offsets),
    ):
        for index, value in enumerate(values or (0,)):
            struct.pack_into("<i", payload, offsets[name] + index * 16, int(value))
    if "uniformBase" in offsets:
        struct.pack_into("<i", payload, offsets["uniformBase"], 0)
    payload = payload[: (cursor + 15) // 16 * 16]
    return bytes(payload)


def _uniform_binding(source):
    import re

    match = re.search(r"@binding\((\d+)\)\s*\nvar<uniform> _Uniforms", source)
    if match is None:
        raise ValueError("Backward WGSL kernel has no uniform binding")
    return int(match.group(1))
