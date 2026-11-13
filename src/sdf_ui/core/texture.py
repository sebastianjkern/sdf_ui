"""Render-node base types, renderer, and materialized output wrappers."""

__docformat__ = "google"

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Self

from numpy.typing import NDArray
from PIL import Image, ImageOps

from ..gpu.context import Context
from ..gpu.nodes.common import build
from ..util import hex_col
from ._suggestions import unknown_name_message


def _texture_bytes(texture):
    read_bytes = getattr(texture, "read_bytes", None)
    return read_bytes() if callable(read_bytes) else texture.read()


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


@dataclass
class RenderStats:
    """Counters accumulated across calls made through one renderer."""

    shader_dispatches: int = 0
    shader_dispatches_by_name: Dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    cache_skips: int = 0
    texture_allocations: int = 0
    texture_allocations_by_kind: Dict[str, int] = field(default_factory=dict)
    render_calls: int = 0
    elapsed_seconds: float = 0.0

    def record_shader_dispatch(self, shader_name, dispatch_groups):
        self.shader_dispatches += 1
        self.shader_dispatches_by_name[shader_name] = (
            self.shader_dispatches_by_name.get(shader_name, 0) + 1
        )

    def record_texture_allocation(self, kind):
        self.texture_allocations += 1
        self.texture_allocations_by_kind[kind] = (
            self.texture_allocations_by_kind.get(kind, 0) + 1
        )

    def as_dict(self):
        return {
            "shader_dispatches": self.shader_dispatches,
            "shader_dispatches_by_name": dict(self.shader_dispatches_by_name),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_writes": self.cache_writes,
            "cache_skips": self.cache_skips,
            "texture_allocations": self.texture_allocations,
            "texture_allocations_by_kind": dict(self.texture_allocations_by_kind),
            "render_calls": self.render_calls,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class RenderCacheInfo:
    """Immutable snapshot of cache size and activity counters."""

    entries: int
    hits: int
    misses: int
    writes: int
    skips: int


class Renderer:
    """Render nodes on one context while reusing a caller-provided cache.

    Use ``Context.session`` to create a renderer tied to a canvas. The
    ``evaluation_rect`` option limits compatible pointwise graphs to a canvas
    region, and ``stats`` and ``cache_info()`` expose work performed.
    """

    def __init__(self, ctx, params=None, cache=None, evaluation_rect=None):
        if ctx is None:
            raise ValueError("A Context is required to render a texture")
        self.ctx = ctx
        self.params = params or {}
        self.cache = {} if cache is None else cache
        self.evaluation_rect = evaluation_rect
        self.stats = RenderStats()
        self._render_depth = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def render(self, value):
        is_root_render = self._render_depth == 0
        if is_root_render:
            start = perf_counter()
            previous_stats = getattr(self.ctx, "_active_render_stats", None)
            previous_rect = self.ctx._active_evaluation_rect
            if (
                self.evaluation_rect is not None
                and isinstance(value, TextureNode)
                and value.tex is not None
            ):
                raise ValueError(
                    "A materialized image cannot be re-rendered into a crop"
                )
            rect = self._resolve_evaluation_rect(value)
            if rect is not None:
                self._validate_region_graph(value)
            self.ctx._active_evaluation_rect = rect
            self.ctx._active_render_stats = self.stats
            self.stats.render_calls += 1

        self._render_depth += 1
        try:
            if isinstance(value, TextureNode) or isinstance(value, MultiOutputResult):
                return self._eval(value)
            return self._resolve(value)
        finally:
            self._render_depth -= 1
            if is_root_render:
                self.stats.elapsed_seconds += perf_counter() - start
                self.ctx._active_render_stats = previous_stats
                self.ctx._active_evaluation_rect = previous_rect
                self.ctx.last_render_stats = self.stats

    def _resolve_evaluation_rect(self, value):
        from ..gpu.regions import normalize_rect

        if self.evaluation_rect is not None:
            return normalize_rect(self.evaluation_rect, self.ctx.size)
        regions = self._materialized_regions(value)
        if len(regions) > 1:
            raise ValueError("Cropped input images must have the same canvas region")
        return normalize_rect(next(iter(regions), None), self.ctx.size)

    def _materialized_regions(self, value, visited=None):
        visited = set() if visited is None else visited
        if id(value) in visited:
            return set()
        visited.add(id(value))
        if isinstance(value, MultiOutputResult):
            children = value.inputs
        elif isinstance(value, TextureNode):
            if value.tex is not None:
                origin = getattr(value.tex, "origin", (0, 0))
                canvas_size = getattr(value.tex, "canvas_size", value.tex.size)
                if origin != (0, 0) or value.tex.size != canvas_size:
                    return {
                        (*origin, *value.tex.size),
                    }
            children = value.inputs
        else:
            return set()
        regions = set()
        for child in children:
            regions.update(self._materialized_regions(child, visited))
        return regions

    def _validate_region_graph(self, value, visited=None):
        from sdf_ui.gpu.nodes.registry import registry

        visited = set() if visited is None else visited
        if id(value) in visited:
            return
        visited.add(id(value))
        children = self._region_children(value, registry)
        for child in children:
            self._validate_region_graph(child, visited)

    @staticmethod
    def _region_children(value, registry):
        if isinstance(value, MultiOutputResult):
            return value.inputs
        if not isinstance(value, TextureNode) or value.tex is not None:
            return ()
        plugin = registry.get(value.op)
        shader = plugin.shader
        if shader is None or shader.differentiation is None:
            if plugin.render_func is not None:
                reason = "host-rendered node"
            else:
                reason = "pointwise shader"
            raise NotImplementedError(
                f"Evaluation rectangles require a {reason} for '{plugin.name}'"
            )
        sampling = shader.differentiation.sampling
        if sampling != "pointwise":
            raise NotImplementedError(
                f"Evaluation rectangles do not yet support '{plugin.name}' sampling: {sampling}"
            )
        return value.inputs

    def close(self) -> None:
        self.cache.clear()

    def clear_cache(self):
        self.cache.clear()

    def cache_info(self):
        return RenderCacheInfo(
            entries=len(self.cache),
            hits=self.stats.cache_hits,
            misses=self.stats.cache_misses,
            writes=self.stats.cache_writes,
            skips=self.stats.cache_skips,
        )

    def _eval(self, node):
        if getattr(node, "tex", None) is not None:
            if getattr(node.tex, "closed", False):
                raise RuntimeError("Cannot use a closed texture node as an input")
            return node

        cache_key = self._node_key(node)
        if node.should_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            if not getattr(getattr(cached, "tex", None), "closed", False):
                self.stats.cache_hits += 1
                return cached
            del self.cache[cache_key]
        if node.should_cache:
            self.stats.cache_misses += 1
        else:
            self.stats.cache_skips += 1

        from sdf_ui.gpu.nodes.registry import registry

        plugin = registry.get(node.op)
        inputs = tuple(self.render(input_) for input_ in node.inputs)
        params = dict((key, self._resolve(value)) for key, value in node.params)
        result = plugin.render(self, inputs, params)

        if node.should_cache:
            self.cache[cache_key] = result
            self.stats.cache_writes += 1

        return result

    def _resolve(self, value):
        if isinstance(value, TextureNode) or isinstance(value, MultiOutputResult):
            return self._eval(value)
        from ..learning.autodiff import Tensor
        from .expressions import Expr, evaluate_expr

        if isinstance(value, (Expr, Tensor, str)):
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

        context_key = (
            id(self.ctx),
            self.ctx.size,
            self.ctx._active_evaluation_rect,
        )
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
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Release the GPU image owned by this materialized texture node.

        Calling ``close`` on an unevaluated render node has no effect. A closed
        materialized node cannot be rendered or read again.
        """
        texture = getattr(self, "tex", None)
        close = getattr(texture, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> Self:
        if self.tex is not None and getattr(self.tex, "closed", False):
            raise RuntimeError("Texture node is closed")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __getattr__(self, name):
        from sdf_ui.gpu.nodes.registry import registry

        method_names = registry.method_names_for(self.kind)
        if name in method_names:
            plugin = registry.get(name)

            def method(*args, **kwargs):
                return build(name, self, *args, **kwargs)

            method.__name__ = name
            method.__qualname__ = f"{type(self).__name__}.{name}"
            method.__doc__ = plugin.docstring(bound=True)
            method.__signature__ = plugin.signature(bound=True)
            return method
        message = unknown_name_message(
            f"{type(self).__name__!s} has no method", name, method_names
        )
        raise AttributeError(message)

    def __dir__(self):
        from sdf_ui.gpu.nodes.registry import registry

        names = set(super().__dir__())
        names.update(registry.method_names_for(self.kind))
        return sorted(names)

    def render(self, ctx=None, params=None, cache=None, evaluation_rect=None) -> Self:
        """Materialize this node, optionally cropping to a canvas pixel rect.

        The rectangle is ``(x, y, width, height)``. All shaders in the render
        tree use the same region, so this currently supports pointwise
        operations and rejects filters that need neighboring pixels.
        """
        if self.tex is not None and getattr(self.tex, "closed", False):
            raise RuntimeError("Cannot render a closed texture node")
        if self.tex is not None and ctx is None:
            if evaluation_rect is not None:
                raise ValueError(
                    "A materialized image cannot be re-rendered into a crop"
                )
            return self
        return Renderer(
            ctx or self.context,
            params=params,
            cache=cache,
            evaluation_rect=evaluation_rect,
        ).render(self)

    def read(self) -> NDArray[Any]:
        """Copy this materialized texture's pixels into a NumPy array.

        SDF textures return a two-dimensional float array. Color textures
        return an ``(height, width, 4)`` array in the image's native dtype.
        """
        if self.tex is None:
            raise RuntimeError(
                "Texture node must be rendered before its pixels can be read"
            )
        if getattr(self.tex, "closed", False):
            raise RuntimeError("Cannot read a closed texture node")
        return self.tex.read()

    @property
    def size(self) -> tuple[int, int]:
        """Return the materialized image's ``(width, height)`` in pixels."""
        return self._image_metadata("size")

    @property
    def origin(self) -> tuple[int, int]:
        """Return the materialized image's top-left canvas-space pixel."""
        return self._image_metadata("origin", (0, 0))

    @property
    def canvas_size(self) -> tuple[int, int]:
        """Return the full canvas size associated with this image."""
        return self._image_metadata("canvas_size", None) or self.size

    def _image_metadata(self, name: str, default: Any = None) -> Any:
        if self.tex is None:
            raise RuntimeError(
                f"Texture node must be rendered before '{name}' is available"
            )
        if getattr(self.tex, "closed", False):
            raise RuntimeError(f"Cannot access '{name}' on a closed texture node")
        return getattr(self.tex, name, default)

    def show(
        self,
        ctx: Any = None,
        params: dict[str, Any] | None = None,
        cache: dict[Any, Any] | None = None,
        conversion: bool = True,
        size: tuple[int, int] | None = None,
        evaluation_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        def _show(texture):
            converted = None
            if (
                getattr(texture, "mode", None) == "LAB"
                and conversion
                and hasattr(texture, "to_rgb")
            ):
                converted = texture.to_rgb().render(texture.context)
                texture = converted
            try:
                ImageOps.flip(Image.fromarray(texture.read())).show()
            finally:
                if converted is not None:
                    converted.close()

        self._with_rendered_texture(
            _show,
            ctx=ctx,
            params=params,
            cache=cache,
            size=size,
            evaluation_rect=evaluation_rect,
        )

    def save(
        self,
        name: str | Path = "./image.png",
        ctx: Any = None,
        params: dict[str, Any] | None = None,
        cache: dict[Any, Any] | None = None,
        conversion: bool = True,
        size: tuple[int, int] | None = None,
        evaluation_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        def _save(texture):
            converted = None
            if self.kind == "color":
                if (
                    getattr(texture, "mode", None) == "LAB"
                    and conversion
                    and hasattr(texture, "to_rgb")
                ):
                    converted = texture.to_rgb().render(texture.context)
                    texture = converted
            try:
                if self.kind == "color":
                    image = Image.frombytes(
                        "RGBA", texture.size, _texture_bytes(texture.tex), "raw"
                    )
                else:
                    image = Image.frombytes(
                        "F", texture.size, _texture_bytes(texture.tex), "raw"
                    )
                    image = image.convert("L")

                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                image.save(name)
            finally:
                if converted is not None:
                    converted.close()

        self._with_rendered_texture(
            _save,
            ctx=ctx,
            params=params,
            cache=cache,
            size=size,
            evaluation_rect=evaluation_rect,
        )

    def named(self, label: str):
        return self._clone(label=label, should_cache=self.should_cache)

    def cache(self, label: Optional[str] = None):
        return self._clone(label=label or self.label, should_cache=True)

    def uncached(self):
        return self._clone(label=self.label, should_cache=False)

    def _render_for_io(self, ctx, params, cache, size, evaluation_rect):
        return self.render(
            ctx,
            params=params,
            cache=cache,
            evaluation_rect=evaluation_rect,
        )

    def _with_rendered_texture(
        self, func, *, ctx, params, cache, size, evaluation_rect
    ):
        if ctx is not None:
            texture = self._render_for_io(ctx, params, cache, size, evaluation_rect)
            try:
                return func(texture)
            finally:
                if texture is not self and self.tex is None:
                    texture.close()

        if size is not None:
            with Context(size) as temp_ctx:
                return func(
                    self._render_for_io(temp_ctx, params, cache, size, evaluation_rect)
                )

        if self.context is None:
            raise ValueError(
                "A Context is required to save or show an unmaterialized texture; "
                "pass ctx= or size= to create a temporary context"
            )
        return func(
            self._render_for_io(self.context, params, cache, size, evaluation_rect)
        )

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
        from sdf_ui.gpu.nodes.registry import registry

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
        from sdf_ui.gpu.nodes.registry import registry

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

    def render(self, ctx, params=None, cache=None, evaluation_rect=None):
        return Renderer(
            ctx,
            params=params,
            cache=cache,
            evaluation_rect=evaluation_rect,
        ).render(self)

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
