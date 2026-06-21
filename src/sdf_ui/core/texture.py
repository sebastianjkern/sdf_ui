"""Render-node base types, renderer, and materialized output wrappers."""

__docformat__ = "google"

from abc import ABC
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from ..util import hex_col
from .context import Context, decrease_tex_registry
from .plugins.common import build
from .texture_utils import show_texture


def freeze(value):
    if isinstance(value, TextureNode) or isinstance(value, MultiOutputResult):
        return value
    if isinstance(value, dict):
        return tuple(sorted((key, freeze(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


def texture_params(**kwargs):
    return tuple(sorted((key, freeze(value)) for key, value in kwargs.items()))


class Renderer:
    def __init__(self, ctx, params=None, cache=None):
        if ctx is None:
            raise ValueError("A Context is required to render a texture")
        self.ctx = ctx
        self.params = params or {}
        self.cache = {} if cache is None else cache

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def render(self, value):
        if isinstance(value, TextureNode) or isinstance(value, MultiOutputResult):
            return self._eval(value)
        return self._resolve(value)

    def close(self):
        self.cache.clear()

    def _eval(self, node):
        if getattr(node, "tex", None) is not None:
            return node

        cache_key = self._node_key(node)
        if node.should_cache and cache_key in self.cache:
            return self.cache[cache_key]

        from sdf_ui.core.plugins.registry import registry

        plugin = registry.get(node.op)
        inputs = tuple(self.render(input_) for input_ in node.inputs)
        params = dict((key, self._resolve(value)) for key, value in node.params)
        result = plugin.render(self, inputs, params)

        if node.should_cache:
            self.cache[cache_key] = result

        return result

    def _resolve(self, value):
        if isinstance(value, TextureNode) or isinstance(value, MultiOutputResult):
            return self._eval(value)
        from .expressions import Expr, evaluate_expr

        if isinstance(value, Expr) or isinstance(value, str):
            value = evaluate_expr(value, self.ctx, self.params)
        if isinstance(value, tuple):
            return tuple(self._resolve(item) for item in value)
        if isinstance(value, list):
            return [self._resolve(item) for item in value]
        return self._resolve_color(value)

    def _resolve_color(self, value):
        if isinstance(value, str) and value.startswith("#"):
            if len(value) == 4:
                value = "#" + "".join(channel * 2 for channel in value[1:])
            return hex_col(value)
        return value

    def _node_key(self, node):
        if getattr(node, "tex", None) is not None:
            return (
                id(node.context),
                node.kind,
                id(node.tex),
                getattr(node, "mode", None),
            )

        context_key = (id(self.ctx), self.ctx.size)
        input_keys = tuple(
            self._node_key(input_)
            if isinstance(input_, TextureNode) or isinstance(input_, MultiOutputResult)
            else self._freeze_runtime(self._resolve(input_))
            for input_ in node.inputs
        )
        param_keys = tuple(
            (key, self._freeze_runtime(self._resolve(value)))
            for key, value in node.params
        )
        return context_key, node.kind, node.op, input_keys, param_keys

    def _freeze_runtime(self, value):
        if isinstance(value, TextureNode) or isinstance(value, MultiOutputResult):
            return self._node_key(value)
        if isinstance(value, dict):
            return tuple(
                sorted((key, self._freeze_runtime(val)) for key, val in value.items())
            )
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_runtime(item) for item in value)
        return value


class TextureNode(ABC):
    kind = None

    def __init__(
        self,
        tex=None,
        context: Optional[Context] = None,
        *,
        op=None,
        inputs=(),
        params=None,
        label=None,
        should_cache=True,
    ):
        self.tex = tex
        self.context = context
        self.op = op
        self.inputs = tuple(inputs)
        self.params = (
            texture_params(**(params or {}))
            if isinstance(params, dict)
            else tuple(params or ())
        )
        self.label = label
        self.should_cache = should_cache

        if tex is not None and context is None:
            raise ValueError("context can't be None for a materialized texture")
        if tex is None and op is None:
            raise ValueError("either tex or op is required")

    def __del__(self):
        if getattr(self, "tex", None) is not None:
            try:
                self.tex.release()
            except Exception:
                pass
            else:
                decrease_tex_registry()

    def __getattr__(self, name):
        from sdf_ui.core.plugins.registry import registry

        if name in registry.method_names_for(self.kind):

            def method(*args, **kwargs):
                return build(name, self, *args, **kwargs)

            return method
        raise AttributeError(f"{type(self).__name__!s} has no attribute '{name}'")

    def __dir__(self):
        from sdf_ui.core.plugins.registry import registry

        names = set(super().__dir__())
        names.update(registry.method_names_for(self.kind))
        return sorted(names)

    def render(self, ctx=None, params=None, cache=None):
        if self.tex is not None and ctx is None:
            return self
        return Renderer(ctx or self.context, params=params, cache=cache).render(self)

    def show(self, ctx=None, params=None, cache=None, conversion=True, size=None):
        def _show(texture):
            if (
                getattr(texture, "mode", None) == "LAB"
                and conversion
                and hasattr(texture, "to_rgb")
            ):
                texture = texture.to_rgb().render(texture.context)
            show_texture(texture.tex)

        self._with_rendered_texture(
            _show, ctx=ctx, params=params, cache=cache, size=size
        )

    def save(
        self,
        name="./image.png",
        ctx=None,
        params=None,
        cache=None,
        conversion=True,
        size=None,
    ):
        def _save(texture):
            if self.kind == "color":
                if (
                    getattr(texture, "mode", None) == "LAB"
                    and conversion
                    and hasattr(texture, "to_rgb")
                ):
                    texture = texture.to_rgb().render(texture.context)
                image = Image.frombytes(
                    "RGBA", texture.tex.size, texture.tex.read(), "raw"
                )
            else:
                image = Image.frombytes(
                    "F", texture.tex.size, texture.tex.read(), "raw"
                )
                image = image.convert("L")

            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image.save(name)

        self._with_rendered_texture(
            _save, ctx=ctx, params=params, cache=cache, size=size
        )

    def named(self, label: str):
        return self._clone(label=label, should_cache=self.should_cache)

    def cache(self, label: Optional[str] = None):
        return self._clone(label=label or self.label, should_cache=True)

    def uncached(self):
        return self._clone(label=self.label, should_cache=False)

    def _check_type(self, obj):
        if not isinstance(obj, TextureNode) or obj.kind != self.kind:
            raise TypeError(f"{obj} should be a {self.kind} texture")

    def _render_for_io(self, ctx, params, cache, size):
        if ctx is None and size is not None:
            ctx = Context(size)
        return self.render(ctx, params=params, cache=cache)

    def _with_rendered_texture(self, func, *, ctx, params, cache, size):
        if ctx is not None or size is None:
            return func(self._render_for_io(ctx, params, cache, size))

        with Context(size) as temp_ctx:
            return func(self.render(temp_ctx, params=params, cache=cache))

    def _clone(self, *, label, should_cache):
        kwargs = {}
        if hasattr(self, "mode"):
            kwargs["mode"] = self.mode
        return type(self)(
            op=self.op,
            inputs=self.inputs,
            params=self.params,
            label=label,
            should_cache=should_cache,
            **kwargs,
        )


class PostNamespace:
    def __init__(self, texture):
        self.texture = texture

    def to_lab(self):
        return self.texture.to_lab()

    def to_rgb(self):
        return self.texture.to_rgb()

    def __getattr__(self, name):
        from sdf_ui.core.plugins.registry import registry

        try:
            plugin = registry.get(name)
        except KeyError as exc:
            raise AttributeError(f"No postprocessing plugin named '{name}'") from exc
        if (
            plugin.family != "postprocessing"
            or self.texture.kind not in plugin.method_of
        ):
            raise AttributeError(f"No postprocessing plugin named '{name}'")

        def method(*args, **kwargs):
            return build(name, self.texture, *args, **kwargs)

        return method

    def __dir__(self):
        from sdf_ui.core.plugins.registry import registry

        names = set(super().__dir__())
        names.update(
            name
            for name in registry.method_names_for(self.texture.kind)
            if registry.get(name).family == "postprocessing"
        )
        names.update({"to_lab", "to_rgb"})
        return sorted(names)


@dataclass(frozen=True)
class MultiOutputResult:
    op: str
    inputs: tuple = ()
    params: tuple = ()
    label: Optional[str] = None
    should_cache: bool = True
    kind: str = "multi_output"

    def render(self, ctx, params=None, cache=None):
        return Renderer(ctx, params=params, cache=cache).render(self)

    def __iter__(self):
        if self.op != "masked_union":
            raise TypeError(f"Multi-output result '{self.op}' is not iterable")

        from sdf_ui.core.color import ColorSpaceMode, ColorTexture
        from sdf_ui.core.sdf import SDFTexture

        yield SDFTexture(
            op="output", inputs=(self,), params={"index": 0, "kind": "sdf"}
        )
        yield ColorTexture(
            op="output",
            inputs=(self,),
            params={"index": 1, "kind": "color"},
            mode=ColorSpaceMode.RGB,
        )
