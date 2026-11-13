"""Native FP32 shader execution, analytic GPU VJPs, and gradient reductions."""

from collections.abc import Mapping
from hashlib import sha256

import numpy as np

from ...backends.opengl.nodes.registry import registry
from ...learning.shader_derivatives import ShaderGradients
from ..base import Image, shape_size
from .glsl import REDUCE


class GPUImage(Image):
    """An owned FP32 image; reading pixels is explicit and counted."""

    def __init__(self, backend, texture, channels):
        self.backend = backend
        self.texture = texture
        self.channels = channels
        self.size = texture.size
        self.shape = (self.size[1], self.size[0]) + (
            () if channels == 1 else (channels,)
        )
        self.closed = False
        backend._image_generation += 1
        self.generation = backend._image_generation

    def read(self):
        if self.closed:
            raise RuntimeError("GPU image is closed")
        self.backend.stats["image_readbacks"] += 1
        return (
            np.frombuffer(self.texture.read(), dtype=np.float32)
            .reshape(self.shape)
            .copy()
        )

    numpy = read

    def clone(self):
        return self.backend.copy_image(self)

    def close(self):
        if not self.closed:
            self.texture.release()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _set(program, name, value):
    if name not in program:
        return
    if isinstance(value, np.ndarray):
        program[name].write(value.tobytes())
    else:
        program[name].value = value


class GPUShaderBackend:
    """Run original shaders and their analytic derivatives on an OpenGL GPU.

    Forward operation sources are compiled with FP32 image formats.
    Derivative bodies use scalar dual arithmetic to compute local VJPs. Input
    adjoints scatter with float CAS atomics; uniform gradients reduce in two
    stages. No reference renderer or finite-difference backward is used.
    """

    def __init__(self, canvas):
        self.canvas = canvas
        self.ctx = getattr(canvas, "_mgl_ctx", canvas)
        if self.ctx.version_code < 430:
            raise RuntimeError(
                "GPU training requires OpenGL 4.3 compute shaders and FP32 image storage"
            )
        self.descriptors = {value.name: value for value in registry.shader_files()}
        shared_programs = getattr(canvas, "_graph_programs", None)
        self._owns_programs = shared_programs is None
        self.programs = {} if shared_programs is None else shared_programs
        self._shader_keys = {}
        self._scratch = {}
        self._forward_local_size = {}
        self._image_generation = 0
        self.status = self.ctx.buffer(reserve=4)
        self.status.clear()
        self.stats = {
            "forward_dispatches": 0,
            "backward_dispatches": 0,
            "reduction_dispatches": 0,
            "image_readbacks": 0,
            "image_uploads": 0,
            "program_compilations": 0,
        }
        self.closed = False

    def _check(self):
        if self.closed:
            raise RuntimeError("GPU backend is closed")

    @property
    def device_info(self):
        return {
            key: self.ctx.info.get(key)
            for key in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")
        }

    def create_buffer(self, size):
        self._check()
        return self.ctx.buffer(reserve=max(4, int(size)))

    def owns_context(self, context):
        return (
            not getattr(context, "_closed", False)
            and getattr(context, "_mgl_ctx", None) is self.ctx
        )

    def create_optimizer(self, parameters, lr):
        from .optim import GPUAdam

        return GPUAdam(self.canvas, parameters, lr=lr)

    def mask_weight_sum(self, image):
        from .loss import mask_weight_sum

        return mask_weight_sum(self, image)

    def seed_pixel_loss(self, *args):
        from .loss import seed_pixel_loss

        return seed_pixel_loss(self, *args)

    def scratch_buffer(self, name, size):
        """Reuse a backend-owned scratch buffer for sequential GPU work."""
        self._check()
        size = max(4, int(size))
        buffer = self._scratch.get(name)
        if buffer is None or buffer.size < size:
            replacement = self.ctx.buffer(reserve=size)
            if buffer is not None:
                buffer.release()
            self._scratch[name] = replacement
            buffer = replacement
        return buffer

    def contract(self, name):
        self._check()
        try:
            contract = self.descriptors[name].differentiation
        except KeyError as error:
            raise ValueError(f"Unknown GPU shader '{name}'") from error
        if contract is None:
            raise ValueError(f"Shader '{name}' has no GPU differentiation contract")
        descriptor = self.descriptors[name]
        if contract.name != name or contract.path != descriptor.path:
            raise ValueError(
                "Shader derivative contract name/path differs from its forward program"
            )
        return contract

    def derivative_code(self, name):
        self.contract(name)
        return self.descriptors[name].operation.backward_source()

    def shader_fingerprints(self, name):
        """Prepare a forward program and fingerprint both shader directions."""
        self._program(name)
        operation = self.descriptors[name].operation
        return (
            sha256(self.derivative_code(name).encode()).hexdigest(),
            sha256(
                operation.forward_source(color_format="rgba32f").encode()
            ).hexdigest(),
        )

    def _program(self, name, backward=False):
        descriptor_key = (name, backward)
        key = self._shader_keys.get(descriptor_key)
        if key is None:
            self.contract(name)
            if backward:
                source = self.derivative_code(name)
            else:
                source = self.descriptors[name].operation.forward_source(
                    color_format="rgba32f"
                )
                self._forward_local_size[name] = self.descriptors[
                    name
                ].operation.workgroup_size(source)
            key = ("$shader", name, backward, sha256(source.encode()).hexdigest())
        else:
            source = None
        if key not in self.programs:
            if source is None:
                raise RuntimeError(
                    "Shader program cache was cleared while backend was active"
                )
            try:
                self.programs[key] = self.ctx.compute_shader(source)
            except Exception as error:
                raise RuntimeError(
                    f"Cannot compile {'backward' if backward else 'FP32 forward'} shader '{name}': {error}"
                ) from error
            self.stats["program_compilations"] += 1
        self._shader_keys[descriptor_key] = key
        return self.programs[key]

    def defaults(self, name):
        contract = self.contract(name)
        return getattr(contract, "defaults", None) or {}

    def uniforms(self, name, params):
        """Validate and snapshot raw uniforms, resetting omitted GLSL values."""
        contract = self.contract(name)
        params = {} if params is None else dict(params)
        names = {argument.name for argument in contract.parameters}
        if set(params) - names:
            raise ValueError(
                f"Unknown uniforms for '{name}': {sorted(set(params) - names)}"
            )
        defaults = self.defaults(name)
        result = {}
        for argument in contract.parameters:
            value = params.get(
                argument.name, defaults.get(argument.name, np.zeros(argument.shape))
            )
            if hasattr(value, "numpy"):
                value = value.numpy()
            array = np.asarray(value, dtype=np.float32)
            if array.shape != argument.shape or not np.all(np.isfinite(array)):
                raise ValueError(
                    f"Uniform '{argument.name}' must have finite shape {argument.shape}"
                )
            result[argument.name] = array.copy()
        return result

    def upload(self, pixels):
        self._check()
        pixels = np.asarray(pixels, dtype=np.float32)
        if pixels.ndim == 2:
            channels = 1
        elif pixels.ndim == 3 and pixels.shape[2] == 4:
            channels = 4
        else:
            raise ValueError(
                "Images must have shape (height,width) or (height,width,4)"
            )
        if not all(pixels.shape[:2]) or not np.all(np.isfinite(pixels)):
            raise ValueError("Image dimensions must be positive and values finite")
        texture = self.ctx.texture(
            (pixels.shape[1], pixels.shape[0]),
            channels,
            np.ascontiguousarray(pixels).tobytes(),
            dtype="f4",
        )
        self.stats["image_uploads"] += 1
        return GPUImage(self, texture, channels)

    def allocate(self, size, channels):
        self._check()
        size = tuple(size)
        if (
            channels not in (1, 4)
            or len(size) != 2
            or any(isinstance(v, bool) or int(v) != v or v <= 0 for v in size)
        ):
            raise ValueError(
                "FP32 images require positive integer dimensions and 1 or 4 channels"
            )
        try:
            texture = self.ctx.texture(size, channels, dtype="f4")
        except Exception as error:
            raise RuntimeError(
                "GPU training requires FP32 image storage; reduced precision fallback is disabled"
            ) from error
        return GPUImage(self, texture, channels)

    def copy_image(self, image):
        """Take an owned GPU snapshot without reading image pixels to the CPU."""
        if image.closed or image.backend.ctx is not self.ctx:
            raise ValueError(
                "Cannot copy an image from a closed or different GPU context"
            )
        result = self.allocate(image.size, image.channels)
        format_ = "r32f" if image.channels == 1 else "rgba32f"
        key = (f"$copy{image.channels}", False)
        try:
            if key not in self.programs:
                source = f"""#version 430
layout(local_size_x=16,local_size_y=16) in;
layout({format_},binding=0) readonly uniform image2D sourceImage;
layout({format_},binding=1) writeonly uniform image2D destination;
void main(){{ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(any(greaterThanEqual(p,imageSize(destination))))return;imageStore(destination,p,imageLoad(sourceImage,p));}}
"""
                self.programs[key] = self.ctx.compute_shader(source)
                self.stats["program_compilations"] += 1
            image.texture.bind_to_image(0, read=True, write=False)
            result.texture.bind_to_image(1, read=False, write=True)
            self.programs[key].run(
                (image.size[0] + 15) // 16, (image.size[1] + 15) // 16, 1
            )
            self.ctx.memory_barrier()
            return result
        except Exception:
            result.close()
            raise

    def import_texture(self, texture, channels):
        """Snapshot an existing normalized/float GL texture to owned FP32 storage."""
        if texture.dtype not in {"f1", "f2", "f4"}:
            raise ValueError(
                "Graph texture inputs must have normalized or floating-point channels"
            )
        result = self.allocate(texture.size, channels)
        key = (f"$import{channels}", False)
        try:
            if key not in self.programs:
                format_ = "r32f" if channels == 1 else "rgba32f"
                source = f"""#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(binding=0) uniform sampler2D sourceTexture;
layout({format_},binding=0) writeonly uniform image2D destination;
void main(){{ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(any(greaterThanEqual(p,imageSize(destination))))return;imageStore(destination,p,texelFetch(sourceTexture,p,0));}}
"""
                self.programs[key] = self.ctx.compute_shader(source)
                self.stats["program_compilations"] += 1
            texture.use(0)
            result.texture.bind_to_image(0, read=False, write=True)
            self.programs[key].run(
                (texture.size[0] + 15) // 16, (texture.size[1] + 15) // 16, 1
            )
            self.ctx.memory_barrier()
            return result
        except BaseException:
            result.close()
            raise

    def display_image(self, image, mode=None):
        """Copy a graph input snapshot to an owned display-format texture."""
        from ...core.color import ColorTexture
        from ...core.sdf import SDFTexture

        self._check()
        if image.closed or image.backend.ctx is not self.ctx:
            raise ValueError(
                "Cannot display an image from a closed or different context"
            )
        source_format = "r32f" if image.channels == 1 else "rgba32f"
        texture = self.canvas.r32f() if image.channels == 1 else self.canvas.rgba8()
        result = (
            SDFTexture(tex=texture, context=self.canvas)
            if image.channels == 1
            else ColorTexture(tex=texture, context=self.canvas, mode=mode or "LAB")
        )
        target_format = (
            self.canvas._sdf_image_format if image.channels == 1 else "rgba8"
        )
        key = (f"$display_{source_format}_{target_format}", False)
        if key not in self.programs:
            source = f"""#version 430
layout(local_size_x=16,local_size_y=16) in;
layout({source_format},binding=0) readonly uniform image2D inputImage;
layout({target_format},binding=1) writeonly uniform image2D outputImage;
void main(){{ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(any(greaterThanEqual(p,imageSize(outputImage))))return;imageStore(outputImage,p,imageLoad(inputImage,p));}}
"""
            self.programs[key] = self.ctx.compute_shader(source)
            self.stats["program_compilations"] += 1
        image.texture.bind_to_image(0, read=True, write=False)
        result.tex.bind_to_image(1, read=False, write=True)
        self.programs[key].run(
            (image.size[0] + 15) // 16, (image.size[1] + 15) // 16, 1
        )
        self.ctx.memory_barrier()
        return result

    def validate_images(self, images):
        """Check FP32 forward values with a GPU flag, without pixel readback."""
        self.status.bind_to_storage_buffer(4)
        for image in images:
            key = (f"$validate{image.channels}", False)
            if key not in self.programs:
                format_ = "r32f" if image.channels == 1 else "rgba32f"
                source = f"""#version 430
layout(local_size_x=64) in;
layout({format_},binding=0) readonly uniform image2D sourceImage;
layout(std430,binding=4) buffer Status{{uint bad;}};
void main(){{ivec2 size=imageSize(sourceImage);int n=int(gl_GlobalInvocationID.x);if(n>=size.x*size.y)return;vec4 v=imageLoad(sourceImage,ivec2(n%size.x,n/size.x));for(int c=0;c<{image.channels};c++)if(isnan(v[c])||isinf(v[c]))atomicOr(bad,4u);}}
"""
                self.programs[key] = self.ctx.compute_shader(source)
                self.stats["program_compilations"] += 1
            image.texture.bind_to_image(0, read=True, write=False)
            self.programs[key].run((image.size[0] * image.size[1] + 63) // 64, 1, 1)
            self.ctx.memory_barrier()

    def forward(self, name, inputs=None, params=None, size=None):
        contract = self.contract(name)
        inputs, size = self._validate_forward_request(name, contract, inputs, size)
        uniforms = self.uniforms(name, params)
        program = self._program(name)
        outputs = tuple(
            self.allocate(size, shape_size(arg.shape)) for arg in contract.outputs
        )
        try:
            self._bind_forward_images(program, contract, inputs, outputs)
            self._write_forward_uniforms(program, contract, uniforms)
            self._dispatch_forward(name, program, size)
        except Exception:
            for image in outputs:
                image.close()
            raise
        return outputs

    def _validate_forward_request(self, name, contract, inputs, size):
        inputs = {} if inputs is None else dict(inputs)
        if set(inputs) != {argument.name for argument in contract.inputs}:
            raise ValueError(f"Shader '{name}' input names differ from its contract")
        if size is None:
            size = (
                next(iter(inputs.values())).size if inputs else tuple(self.canvas.size)
            )
        size = tuple(size)
        if len(size) != 2 or any(int(value) != value or value <= 0 for value in size):
            raise ValueError("GPU image size must be two positive integers")
        self._validate_forward_inputs(name, contract, inputs, size)
        return inputs, size

    def _validate_forward_inputs(self, name, contract, inputs, size):
        for argument in contract.inputs:
            image = inputs[argument.name]
            if image.closed or image.backend.ctx is not self.ctx or image.size != size:
                raise ValueError(f"Incompatible GPU input image '{argument.name}'")
            if image.channels != shape_size(argument.shape):
                raise ValueError(f"Incompatible GPU input image '{argument.name}'")

    @staticmethod
    def _bind_forward_images(program, contract, inputs, outputs):
        location = 0
        for argument, image in zip(contract.outputs, outputs):
            image.texture.bind_to_image(location, read=False, write=True)
            _set(program, argument.name, location)
            location += 1
        for argument in contract.inputs:
            inputs[argument.name].texture.bind_to_image(
                location, read=True, write=False
            )
            _set(program, argument.name, location)
            location += 1

    @staticmethod
    def _write_forward_uniforms(program, contract, uniforms):
        for argument in contract.parameters:
            if argument.name not in program:
                continue
            value = uniforms[argument.name]
            if argument.kind in {"int", "bool"}:
                if np.any(value != np.trunc(value)) or (
                    argument.kind == "bool" and not np.all((value == 0) | (value == 1))
                ):
                    raise ValueError(
                        "Integer/boolean shader uniforms must contain exact control values"
                    )
                value = value.astype(np.int32)
            program[argument.name].write(value.tobytes())

    def _dispatch_forward(self, name, program, size):
        local_x, local_y, _ = self._forward_local_size[name]
        program.run(
            (size[0] + local_x - 1) // local_x,
            (size[1] + local_y - 1) // local_y,
            1,
        )
        self.ctx.memory_barrier()
        self.stats["forward_dispatches"] += 1

    def clear_status(self):
        self.status.clear()

    def check_status(self):
        self.ctx.memory_barrier()
        code = int(np.frombuffer(self.status.read(), dtype=np.uint32)[0])
        if code:
            raise FloatingPointError(
                f"GPU differentiation failed (status={code}: nonfinite value or derivative sample overflow)"
            )

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
        """Accumulate into GPU buffers; only small uniform values cross to GPU."""
        contract = self.contract(name)
        uniforms = self.uniforms(name, params)
        values = (
            np.concatenate([uniforms[a.name].reshape(-1) for a in contract.parameters])
            if contract.parameters
            else np.zeros(1, dtype=np.float32)
        )
        components = sum(shape_size(a.shape) for a in contract.parameters)
        if len(input_offsets) != len(contract.inputs) or len(output_offsets) != len(
            contract.outputs
        ):
            raise ValueError("GPU adjoint offsets differ from the shader's port counts")
        if size is None:
            size = (
                next(iter(inputs.values())).size if inputs else tuple(self.canvas.size)
            )
        groups = (size[0] * size[1] + 63) // 64
        parameter_buffer = self.scratch_buffer("backward_parameters", values.nbytes)
        parameter_buffer.write(values.tobytes())
        partials = self.scratch_buffer("backward_partials", groups * components * 4)
        program = self._program(name, backward=True)
        for index, argument in enumerate(contract.inputs):
            inputs[argument.name].texture.bind_to_image(index, read=True, write=False)
        parameter_buffer.bind_to_storage_buffer(0)
        adjoints.bind_to_storage_buffer(1)
        partials.bind_to_storage_buffer(2)
        self.status.bind_to_storage_buffer(4)
        _set(program, "size", size)
        _set(program, "uniformBase", 0)
        _set(
            program,
            "inputOffsets",
            np.asarray(input_offsets or [0], dtype=np.int32),
        )
        _set(program, "outputOffsets", np.asarray(output_offsets, dtype=np.int32))
        program.run(groups, 1, 1)
        self.ctx.memory_barrier()
        self.stats["backward_dispatches"] += 1
        if components:
            if ("reduce", True) not in self.programs:
                self.programs[("reduce", True)] = self.ctx.compute_shader(REDUCE)
                self.stats["program_compilations"] += 1
            reducer = self.programs[("reduce", True)]
            uniform_gradients.bind_to_storage_buffer(3)
            _set(reducer, "groups", groups)
            _set(reducer, "components", components)
            _set(reducer, "resultOffset", result_offset)
            reducer.run(components, 1, 1)
            self.ctx.memory_barrier()
            self.stats["reduction_dispatches"] += 1

    def vjp(self, name, inputs, params, output_gradients):
        """Standalone GPU VJP with explicit image upload/readback for checking."""
        contract = self.contract(name)
        images = {}
        outputs = ()
        buffers = []
        try:
            self._upload_vjp_inputs(inputs, images)
            outputs = self.forward(name, images, params)
            input_offsets, output_offsets, offset = self._vjp_offsets(
                contract, images, outputs
            )
            adjoints = self.ctx.buffer(reserve=offset * 4)
            buffers.append(adjoints)
            adjoints.clear()
            self._seed_vjp_outputs(
                contract, outputs, output_gradients, output_offsets, adjoints
            )
            n = sum(shape_size(a.shape) for a in contract.parameters)
            gradients = self.ctx.buffer(reserve=max(4, n * 4))
            buffers.append(gradients)
            gradients.clear()
            self.clear_status()
            self.backward_into(
                name, images, params, adjoints, input_offsets, output_offsets, gradients
            )
            self.check_status()
            return self._read_vjp_gradients(
                contract, images, input_offsets, adjoints, gradients
            )
        finally:
            for image in (*images.values(), *outputs):
                image.close()
            for buffer in buffers:
                buffer.release()

    def _upload_vjp_inputs(self, inputs, images):
        for key, value in inputs.items():
            images[key] = self.upload(
                value.numpy() if hasattr(value, "numpy") else value
            )

    @staticmethod
    def _vjp_offsets(contract, images, outputs):
        input_offsets, output_offsets = [], []
        offset = 0
        for argument in contract.inputs:
            input_offsets.append(offset)
            offset += int(np.prod(images[argument.name].shape))
        for image in outputs:
            output_offsets.append(offset)
            offset += int(np.prod(image.shape))
        return input_offsets, output_offsets, offset

    @staticmethod
    def _seed_vjp_outputs(contract, outputs, output_gradients, offsets, adjoints):
        if isinstance(output_gradients, Mapping):
            seeds = [output_gradients[arg.name] for arg in contract.outputs]
        else:
            seeds = [output_gradients] if len(outputs) == 1 else list(output_gradients)
        if len(seeds) != len(outputs):
            raise ValueError("A seed is required for each shader output")
        for seed, image, base in zip(seeds, outputs, offsets):
            seed = np.asarray(seed, dtype=np.float32)
            if seed.shape != image.shape or not np.all(np.isfinite(seed)):
                raise ValueError(
                    "Upstream gradients must be finite and match output shapes"
                )
            adjoints.write(np.ascontiguousarray(seed).tobytes(), offset=base * 4)

    def _read_vjp_gradients(self, contract, images, input_offsets, adjoints, gradients):
        raw = np.frombuffer(adjoints.read(), dtype=np.float32)
        self.stats["image_readbacks"] += 1
        input_grads = {
            argument.name: raw[
                base : base + int(np.prod(images[argument.name].shape))
            ].reshape(images[argument.name].shape)
            for argument, base in zip(contract.inputs, input_offsets)
        }
        raw = np.frombuffer(gradients.read(), dtype=np.float32)
        parameter_grads, base = {}, 0
        for argument in contract.parameters:
            count = shape_size(argument.shape)
            parameter_grads[argument.name] = raw[base : base + count].reshape(
                argument.shape
            )
            base += count
        return ShaderGradients(input_grads, parameter_grads)

    def close(self):
        if not self.closed:
            if self._owns_programs:
                for program in self.programs.values():
                    program.release()
                self.programs.clear()
            for buffer in self._scratch.values():
                buffer.release()
            self._scratch.clear()
            self.status.release()
            self.closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
