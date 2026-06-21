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
class ShaderFile:
    name: str
    path: str


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

    def bind(self, args, kwargs):
        if len(args) < len(self.input_kinds):
            raise TypeError(f"{self.name} expects at least {len(self.input_kinds)} input argument(s)")

        inputs = args[:len(self.input_kinds)]
        param_args = args[len(self.input_kinds):]
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
        from sdf_ui.core.operations import run_shader
        from sdf_ui.core.sdf import SDFTexture

        ctx = renderer.ctx
        if self.result == TextureKind.SDF:
            tex = ctx.r32f()
        elif self.result == TextureKind.COLOR:
            tex = ctx.rgba8()
        else:
            raise ValueError(f"Plugin '{self.name}' cannot use the default renderer for result '{self.result}'")

        uniforms = {"destTex": 0}
        if self.make_uniforms is not None:
            uniforms.update(self.make_uniforms(params))

        image_bindings = [(tex, 0, False, True)]
        for location, (uniform_name, texture) in enumerate(zip(self.input_uniforms, inputs), start=1):
            uniforms[uniform_name] = location
            image_bindings.append((texture.tex, location, True, False))

        run_shader(ctx, self.shader.name, uniforms=uniforms, image_bindings=image_bindings)

        if self.result == TextureKind.SDF:
            return SDFTexture(tex=tex, context=ctx)
        mode = self.mode
        if mode is None and inputs and hasattr(inputs[0], "mode"):
            mode = inputs[0].mode
        return ColorTexture(tex=tex, context=ctx, mode=mode or ColorSpaceMode.LAB)
