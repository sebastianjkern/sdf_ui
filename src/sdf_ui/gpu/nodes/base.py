__docformat__ = "google"

import inspect
from dataclasses import dataclass
from typing import Callable, Optional

from ...core._suggestions import unknown_name_message


class TextureKind:
    SDF = "sdf"
    COLOR = "color"
    MULTI_OUTPUT = "multi_output"


class PluginFamily:
    PRIMITIVE = "primitive"
    SHADING = "shading"
    LAYER = "layer"
    POSTPROCESSING = "postprocessing"


@dataclass(frozen=True)
class ImageBinding:
    """A shader image binding and its resource slot."""

    name: str
    unit: int
    read: bool = False
    write: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Image binding needs a uniform name")
        if (
            isinstance(self.unit, bool)
            or not isinstance(self.unit, int)
            or self.unit < 0
        ):
            raise ValueError("Image binding unit must be a nonnegative integer")
        if not self.read and not self.write:
            raise ValueError("Image binding must be readable or writable")


@dataclass(frozen=True)
class Uniform:
    """A named shader parameter and its GLSL value type."""

    name: str
    glsl_type: str


@dataclass(frozen=True)
class ShaderInterface:
    """Shader-owned image layout and parameter declarations."""

    output: ImageBinding
    inputs: tuple = ()
    uniforms: tuple = ()
    extra_outputs: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "uniforms", tuple(self.uniforms))
        object.__setattr__(self, "extra_outputs", tuple(self.extra_outputs))
        images = (self.output, *self.extra_outputs, *self.inputs)
        if len({item.unit for item in images}) != len(images):
            raise ValueError("Shader image units must be unique")
        names = [item.name for item in (*images, *self.uniforms)]
        if len(set(names)) != len(names):
            raise ValueError("Shader uniform names must be unique")
        if any(not item.write for item in (self.output, *self.extra_outputs)) or any(
            not item.read for item in self.inputs
        ):
            raise ValueError("Shader output must be writable and inputs readable")


@dataclass(frozen=True)
class ShaderFile:
    name: str
    path: str
    derivative_contract: object = None
    gpu_derivative: Optional[str] = None
    program: object = None
    interface: Optional[ShaderInterface] = None

    @property
    def operation(self):
        from ..shader_operation import ShaderOperation

        return ShaderOperation(
            self.name,
            self.path,
            self.differentiation,
            self.gpu_derivative,
            self.program,
        )

    @property
    def differentiation(self):
        """Return only the contract declared by this shader extension."""
        return self.derivative_contract

    def backward(self, size, inputs, params, output_gradients):
        """Evaluate the shader's reference VJP at named texture/uniform values."""
        contract = self.differentiation
        if contract is None:
            raise ValueError(f"Shader '{self.name}' has no differentiation contract")
        return contract.vjp(size, inputs, params, output_gradients)


@dataclass(frozen=True)
class Plugin:
    name: str
    family: str
    result: str
    input_kinds: tuple = ()
    params: tuple = ()
    defaults: Optional[dict] = None
    shader: Optional[ShaderFile] = None
    extra_shaders: tuple = ()
    input_uniforms: tuple = ()
    make_uniforms: Optional[Callable] = None
    mode: Optional[str] = None
    public: bool = False
    method_of: tuple = ()
    render_func: Optional[Callable] = None
    # Optional floating-point implementation. Building it from learning.Tensor
    # operations records its backward pass without changing the GPU renderer.
    differentiable_func: Optional[Callable] = None
    # Return one Port or a tuple of Ports using the shared graph builder.
    lowering_func: Optional[Callable] = None

    def method_input_names(self) -> tuple[str, ...]:
        """Return names for texture inputs after a method's bound receiver."""
        extra_inputs = max(0, len(self.input_kinds) - 1)
        if extra_inputs == 0:
            return ()
        if self.name == "fill_from_texture":
            return ("layer",)[:extra_inputs]
        if self.name in {"mask", "layer_mask"}:
            return ("top", "mask")[:extra_inputs]
        if extra_inputs == 1:
            return ("other",)
        return tuple(f"input_{index + 1}" for index in range(extra_inputs))

    def signature(self, *, bound: bool = False) -> inspect.Signature:
        """Return the public callable signature represented by this plugin."""
        parameters = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in (self.method_input_names() if bound else ())
        ]
        defaults = self.defaults or {}
        parameters.extend(
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=defaults.get(name, inspect.Parameter.empty),
            )
            for name in self.params
        )
        return inspect.Signature(parameters)

    def docstring(self, *, bound: bool = False) -> str:
        """Describe the callable signature for factories and bound methods."""
        signature = f"{self.name}{self.signature(bound=bound)}"
        kind_labels = {
            TextureKind.SDF: "SDF",
            TextureKind.COLOR: "color",
            TextureKind.MULTI_OUTPUT: "multi-output",
        }
        if bound:
            kind = kind_labels.get(self.input_kinds[0], self.input_kinds[0])
            action = f"Apply '{self.name}' to this {kind} texture."
        else:
            kind = kind_labels.get(self.result, self.result)
            action = f"Create a {kind} texture with '{self.name}'."
        return f"{action}\n\nCall signature: {signature}."

    def bind(self, args, kwargs):
        kwargs = dict(kwargs)
        input_names = self.method_input_names()
        inputs = []
        for index in range(len(self.input_kinds)):
            name = input_names[index - 1] if index > 0 and input_names else None
            if index < len(args):
                if name is not None and name in kwargs:
                    raise TypeError(f"{self.name} got multiple values for '{name}'")
                inputs.append(args[index])
            elif name is not None and name in kwargs:
                inputs.append(kwargs.pop(name))
            elif index == 0 or name is None:
                raise TypeError(
                    f"{self.name} expects at least {len(self.input_kinds)} input argument(s)"
                )
            else:
                unknown = next(
                    (
                        key
                        for key in kwargs
                        if key not in self.params and key not in input_names
                    ),
                    None,
                )
                if unknown is not None:
                    message = unknown_name_message(
                        f"{self.name} got an unexpected argument",
                        unknown,
                        (*input_names, *self.params),
                    )
                    raise TypeError(message)
                raise TypeError(f"{self.name} missing required input argument '{name}'")

        param_args = args[len(self.input_kinds) :]
        if len(param_args) > len(self.params):
            raise TypeError(f"{self.name} got too many positional arguments")

        params = dict(zip(self.params, param_args))
        for key, value in kwargs.items():
            if key not in self.params:
                message = unknown_name_message(
                    f"{self.name} got an unexpected parameter", key, self.params
                )
                raise TypeError(message)
            if key in params:
                raise TypeError(f"{self.name} got multiple values for '{key}'")
            params[key] = value

        params = {**(self.defaults or {}), **params}
        missing = tuple(param for param in self.params if param not in params)
        if missing:
            names = ", ".join(missing)
            raise TypeError(f"{self.name} missing required parameter(s): {names}")

        return tuple(inputs), params

    def render(self, renderer, inputs, params):
        if self.render_func is not None:
            return self.render_func(renderer, inputs, params)

        if self.shader is None:
            raise ValueError(f"Plugin '{self.name}' has no shader or render function")

        from sdf_ui.core.color import ColorSpaceMode, ColorTexture
        from sdf_ui.core.sdf import SDFTexture

        return self._render_gpu(
            renderer.ctx, inputs, params, ColorSpaceMode, ColorTexture, SDFTexture
        )

    def _render_gpu(
        self, ctx, inputs, params, ColorSpaceMode, ColorTexture, SDFTexture
    ):
        if not ctx.supports_shader(self.shader.name):
            raise NotImplementedError(
                f"WebGPU shader '{self.shader.name}' has not been ported to WGSL"
            )
        interface = self.shader.interface
        if interface is None:
            raise ValueError(
                f"Shader extension '{self.shader.name}' must declare its image and uniform interface"
            )
        if len(interface.inputs) != len(inputs):
            raise ValueError(f"Shader '{self.shader.name}' has invalid input count")

        if self.result == TextureKind.SDF:
            texture = ctx.r32f()
        elif self.result == TextureKind.COLOR:
            texture = ctx.rgba8()
        else:
            raise ValueError(
                f"Plugin '{self.name}' cannot render result '{self.result}' on WebGPU"
            )
        uniforms = {} if self.make_uniforms is None else self.make_uniforms(params)
        images = tuple(item.tex for item in inputs)
        try:
            ctx.run_node_shader(self.shader, texture, images, uniforms)
        except BaseException:
            texture.close()
            raise

        if self.result == TextureKind.SDF:
            return SDFTexture(tex=texture, context=ctx)
        mode = self.mode
        if mode is None and inputs and hasattr(inputs[0], "mode"):
            mode = inputs[0].mode
        return ColorTexture(tex=texture, context=ctx, mode=mode or ColorSpaceMode.LAB)
