__docformat__ = "google"

from dataclasses import dataclass
from typing import Callable, Optional


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
    """An extension-owned image uniform and its OpenGL image unit."""

    name: str
    unit: int
    read: bool = False
    write: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Image binding needs a uniform name")
        if isinstance(self.unit, bool) or not isinstance(self.unit, int) or self.unit < 0:
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
        if (any(not item.write for item in (self.output, *self.extra_outputs))
                or any(not item.read for item in self.inputs)):
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

    def bind(self, args, kwargs):
        if len(args) < len(self.input_kinds):
            raise TypeError(
                f"{self.name} expects at least {len(self.input_kinds)} input argument(s)"
            )

        inputs = args[: len(self.input_kinds)]
        param_args = args[len(self.input_kinds) :]
        if len(param_args) > len(self.params):
            raise TypeError(f"{self.name} got too many positional arguments")

        params = dict(zip(self.params, param_args))
        for key, value in kwargs.items():
            if key not in self.params:
                raise TypeError(f"{self.name} got an unexpected parameter '{key}'")
            if key in params:
                raise TypeError(f"{self.name} got multiple values for '{key}'")
            params[key] = value

        params = {**(self.defaults or {}), **params}
        missing = tuple(param for param in self.params if param not in params)
        if missing:
            names = ", ".join(missing)
            raise TypeError(f"{self.name} missing required parameter(s): {names}")

        return inputs, params

    def render(self, renderer, inputs, params):
        if self.render_func is not None:
            return self.render_func(renderer, inputs, params)

        if self.shader is None:
            raise ValueError(f"Plugin '{self.name}' has no shader or render function")

        from sdf_ui.core.color import ColorSpaceMode, ColorTexture
        from sdf_ui.backends.opengl.operations import run_shader
        from sdf_ui.core.sdf import SDFTexture

        ctx = renderer.ctx
        if self.result == TextureKind.SDF:
            tex = ctx.r32f()
        elif self.result == TextureKind.COLOR:
            tex = ctx.rgba8()
        else:
            raise ValueError(
                f"Plugin '{self.name}' cannot use the default renderer for result '{self.result}'"
            )

        interface = self.shader.interface
        if interface is None:
            raise ValueError(
                f"Shader extension '{self.shader.name}' must declare its image and uniform interface"
            )
        if len(interface.inputs) != len(inputs):
            raise ValueError(f"Shader '{self.shader.name}' has invalid input count")
        output = interface.output
        image_inputs = interface.inputs
        uniforms = {output.name: output.unit}
        if self.make_uniforms is not None:
            uniforms.update(self.make_uniforms(params))

        image_bindings = [(tex, output.unit, output.read, output.write)]
        for binding, texture in zip(image_inputs, inputs):
            uniforms[binding.name] = binding.unit
            image_bindings.append((texture.tex, binding.unit, binding.read, binding.write))

        run_shader(
            ctx, self.shader.name, uniforms=uniforms, image_bindings=image_bindings
        )

        if self.result == TextureKind.SDF:
            return SDFTexture(tex=tex, context=ctx)
        mode = self.mode
        if mode is None and inputs and hasattr(inputs[0], "mode"):
            mode = inputs[0].mode
        return ColorTexture(tex=tex, context=ctx, mode=mode or ColorSpaceMode.LAB)
