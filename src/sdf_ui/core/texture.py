"""Render-node base types, renderer, and materialized output wrappers."""

__docformat__ = "google"

import weakref
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

_CLOSED_TEXTURE = "closed"


def _resolve_canvas_alias(ctx, canvas):
    if canvas is not None:
        if ctx is not None and ctx is not canvas:
            raise ValueError("Pass either ctx or canvas, not both")
        return canvas
    return ctx


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
    """Counters and optional GPU timings accumulated by one renderer.

    GPU timings are collected only when the context was created with
    ``enable_gpu_timing=True`` and the adapter supports timestamp queries.
    """

    shader_dispatches: int = 0
    shader_dispatches_by_name: Dict[str, int] = field(default_factory=dict)
    shader_gpu_seconds_by_name: Dict[str, float] = field(default_factory=dict)
    gpu_timing_supported: bool = False
    _pending_gpu_timestamps: list = field(default_factory=list, repr=False)
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    cache_skips: int = 0
    texture_allocations: int = 0
    texture_allocations_by_kind: Dict[str, int] = field(default_factory=dict)
    texture_reuses: int = 0
    render_calls: int = 0
    elapsed_seconds: float = 0.0
    dirty_layer_hits: int = 0
    dirty_layer_updates: int = 0
    dirty_region_pixels: int = 0
    dirty_region_fallbacks: int = 0

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
            "shader_gpu_seconds_by_name": dict(self.shader_gpu_seconds_by_name),
            "gpu_timing_supported": self.gpu_timing_supported,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_writes": self.cache_writes,
            "cache_skips": self.cache_skips,
            "texture_allocations": self.texture_allocations,
            "texture_allocations_by_kind": dict(self.texture_allocations_by_kind),
            "texture_reuses": self.texture_reuses,
            "render_calls": self.render_calls,
            "elapsed_seconds": self.elapsed_seconds,
            "dirty_layer_hits": self.dirty_layer_hits,
            "dirty_layer_updates": self.dirty_layer_updates,
            "dirty_region_pixels": self.dirty_region_pixels,
            "dirty_region_fallbacks": self.dirty_region_fallbacks,
        }


@dataclass(frozen=True)
class RenderCacheInfo:
    """Immutable snapshot of cache size and activity counters."""

    entries: int
    hits: int
    misses: int
    writes: int
    skips: int


class _FullDirtyFrame(Exception):
    """Signal that an automatic crop cannot be proven safe."""


def _union_bounds(bounds, canvas_size):
    bounds = [item for item in bounds if item is not None]
    if not bounds:
        return None
    x0 = min(item[0] for item in bounds)
    y0 = min(item[1] for item in bounds)
    x1 = max(item[0] + item[2] for item in bounds)
    y1 = max(item[1] + item[3] for item in bounds)
    width, height = canvas_size
    x0, y0 = max(0, int(x0) - 2), max(0, int(y0) - 2)
    x1, y1 = min(width, int(x1 + 999 / 1000) + 2), min(height, int(y1 + 999 / 1000) + 2)
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1 - x0, y1 - y0)


def _primitive_bounds(op, params, canvas_size):
    center = params.get("center", params.get("offset"))
    if center is None:
        return None
    x, y = center
    if op in {"circle", "ring", "ngon"}:
        radius = abs(float(params.get("radius", 0)))
        extent = radius + abs(float(params.get("thickness", 0))) / 2
        rx = ry = extent
    elif op == "ellipse":
        rx, ry = (abs(float(value)) for value in params.get("radii", params.get("size", (0, 0))))
    else:
        rx, ry = (abs(float(value)) for value in params.get("size", (0, 0)))
        radius = params.get("corner_radius", 0)
        if isinstance(radius, (int, float)):
            rx += abs(radius)
            ry += abs(radius)
    if abs(float(params.get("angle", 0))):
        import math
        rx = ry = math.hypot(rx, ry)
    return _union_bounds(((x - rx, y - ry, 2 * rx, 2 * ry),), canvas_size)


def _capsule_bounds(params, canvas_size):
    a, b = params.get("a"), params.get("b")
    if a is None or b is None:
        return None
    radius = abs(float(params.get("radius", params.get("thickness", 0))))
    bounds = (min(a[0], b[0]) - radius, min(a[1], b[1]) - radius,
              abs(a[0] - b[0]) + 2 * radius, abs(a[1] - b[1]) + 2 * radius)
    return _union_bounds((bounds,), canvas_size)


def _inflated_child_bounds(params, child_bounds, canvas_size):
    if not child_bounds or child_bounds[0] is None:
        return None
    x, y, width, height = child_bounds[0]
    margin = max(abs(float(params.get(key, 0) or 0)) for key in ("inflate", "inner", "outer")) + 2
    expanded = (x - margin, y - margin, width + 2 * margin, height + 2 * margin)
    return _union_bounds((expanded,), canvas_size)


def _transform_point(op, params, point):
    import math
    px, py = point
    cx, cy = params.get("center", (0.0, 0.0))
    if op == "translate":
        ox, oy = params.get("offset", (0.0, 0.0))
        return px + ox, py + oy
    if op == "scale":
        factor = params.get("factor", 1.0)
        return cx + (px - cx) * factor, cy + (py - cy) * factor
    angle = params.get("angle", 0.0)
    dx, dy = px - cx, py - cy
    return cx + dx * math.cos(angle) - dy * math.sin(angle), cy + dx * math.sin(angle) + dy * math.cos(angle)


def _transform_bounds(op, params, child_bounds, canvas_size):
    if not child_bounds or child_bounds[0] is None:
        return None
    x, y, width, height = child_bounds[0]
    corners = ((x, y), (x + width, y), (x, y + height), (x + width, y + height))
    transformed = (_transform_point(op, params, point) for point in corners)
    xs, ys = zip(*transformed)
    bounds = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    return _union_bounds((bounds,), canvas_size)


def _node_bounds(op, params, inputs, canvas_size, child_bounds=()):
    """Return conservative pixel bounds for common finite SDF primitives."""
    if op in {"circle", "ring", "ellipse", "rect", "rounded_rect", "ngon"}:
        return _primitive_bounds(op, params, canvas_size)
    if op in {"capsule", "segment"}:
        return _capsule_bounds(params, canvas_size)
    if op in {"fill", "outline"}:
        return _inflated_child_bounds(params, child_bounds, canvas_size)
    if op in {"union", "smooth_union", "intersection", "subtract", "smooth_subtract", "smooth_intersection", "alpha_overlay", "over", "isolines", "transparency", "layer_mask", "to_rgb", "to_lab"}:
        selected = child_bounds[:1] if op in {"isolines", "transparency", "to_rgb", "to_lab"} else child_bounds
        return _union_bounds(selected, canvas_size)
    if op in {"translate", "rotate", "scale"}:
        return _transform_bounds(op, params, child_bounds, canvas_size)
    return None


_SPATIAL_PARAMS = {
    "center", "radius", "r", "radii", "a", "b", "c", "point",
    "point_1", "point_2", "point_3", "start", "end", "points",
    "vertices", "size", "width", "height", "offset", "ox", "oy",
    "box", "scale", "inflate", "inner", "outer", "line_width",
    "thickness", "corner_radius", "k", "s",
    "feather", "edge_width", "spacing", "distance", "blur", "base", "first_line_indent",
    "decoration_thickness", "word_spacing", "letter_spacing", "min_render_size",
}
_SPATIAL_PARAMS_BY_OPERATION = {
    "hatch_fill": {"origin_x", "origin_y", "phase"},
    "isolines": {"phase"},
}


def _scale_spatial_params(op, params, scale):
    """Scale scene-space measurements while leaving colors and quality controls alone."""

    def scaled(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value * scale
        if isinstance(value, tuple):
            return tuple(scaled(item) for item in value)
        if isinstance(value, list):
            return [scaled(item) for item in value]
        return value

    result = dict(params)
    spatial = _SPATIAL_PARAMS | _SPATIAL_PARAMS_BY_OPERATION.get(op, set())
    for key in spatial.intersection(result):
        if op == "blur" and key == "base":
            continue
        result[key] = scaled(result[key])
    if op == "text":

        def scale_span(value):
            if isinstance(value, dict):
                return {
                    key: scaled(item) if key in _SPATIAL_PARAMS else scale_span(item)
                    for key, item in value.items()
                }
            if isinstance(value, (tuple, list)):
                return type(value)(scale_span(item) for item in value)
            return value

        if "value" in result:
            result["value"] = scale_span(result["value"])
    return result


class Renderer:
    """Render nodes on one context while reusing a caller-provided cache.

    Use ``Context.session`` to create a renderer tied to a canvas. The
    ``evaluation_rect`` option limits compatible pointwise graphs to a canvas
    region. Uncached intermediate outputs are returned to the context image pool
    after their final graph use. ``stats`` and ``cache_info()`` expose work.
    """

    def __init__(
        self,
        ctx,
        params=None,
        cache=None,
        evaluation_rect=None,
        progress=0.0,
        dirty_regions=False,
    ):
        if ctx is None:
            raise ValueError("A Context is required to render a texture")
        if dirty_regions and evaluation_rect is not None:
            raise ValueError(
                "dirty_regions cannot be combined with evaluation_rect"
            )
        self.ctx = ctx
        self.params = params or {}
        self.cache = {} if cache is None else cache
        self.evaluation_rect = evaluation_rect
        from .expressions import validate_progress

        self.progress = validate_progress(progress)
        self.stats = RenderStats()
        self._render_depth = 0
        self._graph_ref_counts = {}
        self._graph_remaining_refs = {}
        self._graph_outputs = {}
        self._root_graph_id = None
        self._dirty_regions_enabled = bool(dirty_regions)
        self._dirty_layer_cache = {}
        self._dirty_region_temporaries = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def size(self):
        return self.ctx.size

    @property
    def inputs(self):
        return getattr(self.ctx, "inputs", {})

    def percent(self, value):
        return self.ctx.percent(value)

    def percent_x(self, value):
        return self.ctx.percent_x(value)

    def percent_y(self, value):
        return self.ctx.percent_y(value)

    def percent_of_min(self, value):
        return self.ctx.percent_of_min(value)

    def render(self, value, *, progress=None, dirty_rect=None):
        is_root_render = self._render_depth == 0
        root_state = self._begin_root_render(value, progress, dirty_rect) if is_root_render else None

        self._render_depth += 1
        try:
            if isinstance(value, TextureNode) or isinstance(value, MultiOutputResult):
                if is_root_render and self._dirty_regions_enabled:
                    return self._eval_dirty_layer(value)
                result = self._eval(value)
                if is_root_render:
                    self._release_unused_graph_outputs(result)
                return result
            return self._resolve(value)
        finally:
            self._render_depth -= 1
            if is_root_render:
                self._finish_root_render(root_state)

    def _begin_root_render(self, value, progress, dirty_rect):
        from ..gpu.regions import normalize_rect
        from .expressions import validate_progress
        from .spatial import spatial_scene
        if progress is not None:
            self.progress = validate_progress(progress)
        if dirty_rect is not None and self.evaluation_rect is not None:
            raise ValueError("dirty_rect cannot be combined with evaluation_rect")
        spatial = isinstance(value, (TextureNode, MultiOutputResult)) and spatial_scene(value)
        previous_dirty, previous_cache = self._dirty_regions_enabled, self.cache
        rect = self._root_render_rect(value, spatial, dirty_rect, normalize_rect)
        self._prepare_root_graph(value)
        start = perf_counter()
        previous_stats = getattr(self.ctx, "_active_render_stats", None)
        previous_rect = self.ctx._active_evaluation_rect
        self._validate_root_region(value)
        if spatial:
            self.cache = {}
            self._dirty_regions_enabled = False
        else:
            self._dirty_regions_enabled |= dirty_rect is not None
        self.ctx._active_evaluation_rect = rect
        self.ctx._active_render_stats = self.stats
        self.stats.gpu_timing_supported = self.ctx.gpu_timing_supported
        self.stats.render_calls += 1
        self.ctx._begin_command_batch()
        return spatial, previous_dirty, previous_cache, start, previous_stats, previous_rect

    def _root_render_rect(self, value, spatial, dirty_rect, normalize_rect):
        automatic = not spatial and (
            dirty_rect == "auto" or (dirty_rect is None and self._dirty_regions_enabled)
        )
        if automatic:
            return self._resolve_dirty_rect(value)
        if dirty_rect is not None and not spatial:
            return normalize_rect(dirty_rect, self.ctx.size)
        return self._resolve_evaluation_rect(value)

    def _prepare_root_graph(self, value):
        self._graph_ref_counts = self._count_graph_references(value)
        self._graph_remaining_refs = dict(self._graph_ref_counts)
        self._graph_outputs.clear()
        self._root_graph_id = id(value)

    def _validate_root_region(self, value):
        if self.evaluation_rect is not None and isinstance(value, TextureNode) and value.tex is not None:
            raise ValueError("A materialized image cannot be re-rendered into a crop")
        if self.evaluation_rect is not None:
            self._validate_region_graph(value)

    def _finish_root_render(self, state):
        spatial, previous_dirty, previous_cache, start, previous_stats, previous_rect = state
        self.ctx._end_command_batch()
        self._release_unused_graph_outputs(None)
        for temporary in self._dirty_region_temporaries:
            temporary.close()
        self._dirty_region_temporaries.clear()
        self.ctx._resolve_shader_timings(self.stats)
        self.stats.elapsed_seconds += perf_counter() - start
        self.ctx._active_render_stats = previous_stats
        self.ctx._active_evaluation_rect = previous_rect
        self.ctx.last_render_stats = self.stats
        if spatial:
            self.cache.clear()
            self.cache = previous_cache
            self._dirty_regions_enabled = previous_dirty

    def _resolve_evaluation_rect(self, value):
        from ..gpu.regions import normalize_rect

        if self.evaluation_rect is not None:
            return normalize_rect(self.evaluation_rect, self.ctx.size)
        regions = self._materialized_regions(value)
        if len(regions) > 1:
            raise ValueError("Cropped input images must have the same canvas region")
        return normalize_rect(next(iter(regions), None), self.ctx.size)

    @staticmethod
    def _count_graph_references(root):
        from .spatial import samples

        counts = {}
        visited = set()
        pending = [root]
        while pending:
            value = pending.pop()
            Renderer._count_references_from(value, pending, visited, counts, samples)
        return counts

    @staticmethod
    def _count_references_from(value, pending, visited, counts, samples):
        if not isinstance(value, (TextureNode, MultiOutputResult)) or id(value) in visited:
            return
        if isinstance(value, TextureNode) and value.tex is not None:
            return
        mapped = (binding.args[0] for _, parameter in value.params for binding in samples(parameter))
        for child in (*value.inputs, *mapped):
            if not isinstance(child, (TextureNode, MultiOutputResult)):
                continue
            identity = id(child)
            counts[identity] = counts.get(identity, 0) + 1
            if identity not in visited:
                pending.append(child)
        visited.add(id(value))

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
        from .spatial import is_spatial_node

        if is_spatial_node(value):
            from ..gpu.spatial import tile_dependencies

            if not getattr(self.ctx, "_tiled_export", False):
                raise NotImplementedError(
                    "Use save_tiled for spatial parameter evaluation in regions"
                )
            for child, _ in tile_dependencies(self, value):
                self._validate_region_graph(child, visited)
            return
        children = self._region_children(
            value, registry, tiled=getattr(self.ctx, "_tiled_export", False)
        )
        for child in children:
            self._validate_region_graph(child, visited)

    @staticmethod
    def _region_children(value, registry, *, tiled=False):
        if isinstance(value, TextureNode) and value.tex is not None:
            return ()
        if not isinstance(value, (TextureNode, MultiOutputResult)):
            return ()
        if isinstance(value, TextureNode) and value.op == "output":
            return value.inputs
        plugin = registry.get(value.op)
        shader = plugin.shader
        if Renderer._needs_full_region_inputs(value, shader, tiled):
            return value.inputs
        Renderer._validate_region_shader(plugin, shader)
        return value.inputs

    @staticmethod
    def _needs_full_region_inputs(value, shader, tiled):
        if value.op == "output":
            return True
        if not tiled:
            return False
        from ..gpu.tiled import COMPOSITES, NEIGHBOR_SHADERS
        return value.op in COMPOSITES or (shader is not None and shader.name in NEIGHBOR_SHADERS)

    @staticmethod
    def _validate_region_shader(plugin, shader):
        if shader is None or shader.differentiation is None:
            reason = "host-rendered node" if plugin.render_func is not None else "pointwise shader"
            raise NotImplementedError(
                f"Evaluation rectangles require a {reason} for '{plugin.name}'"
            )
        sampling = shader.differentiation.sampling
        if sampling != "pointwise":
            raise NotImplementedError(
                f"Evaluation rectangles do not yet support '{plugin.name}' sampling: {sampling}"
            )

    def close(self) -> None:
        self._clear_caches()

    def _clear_caches(self):
        self.cache.clear()
        self._dirty_layer_cache.clear()
        self._dirty_region_temporaries.clear()

    def clear_cache(self):
        self._clear_caches()

    def cache_info(self):
        return RenderCacheInfo(
            entries=len(self.cache),
            hits=self.stats.cache_hits,
            misses=self.stats.cache_misses,
            writes=self.stats.cache_writes,
            skips=self.stats.cache_skips,
        )

    def _cache_node(self, node):
        if not node.should_cache:
            return False
        if getattr(self.ctx, "_tiled_export", False):
            return self._graph_ref_counts.get(id(node), 0) > 1
        return True

    def _eval(self, node):
        if self._dirty_regions_enabled:
            return self._eval_dirty_layer(node)
        materialized = self._materialized_input(node)
        if materialized is not None:
            return materialized

        cacheable = self._cache_node(node)
        cache_key = self._node_key(node) if cacheable else None
        cached = self._cached_node(node, cacheable, cache_key)
        if cached is not None:
            return cached
        return self._render_node(node, cacheable, cache_key)

    @staticmethod
    def _materialized_input(node):
        if getattr(node, "tex", None) is None:
            return None
        if getattr(node.tex, _CLOSED_TEXTURE, False):
            raise RuntimeError("Cannot use a closed texture node as an input")
        return node

    def _cached_node(self, node, cacheable, cache_key):
        if cacheable and cache_key in self.cache:
            cached = self.cache[cache_key]
            if not getattr(getattr(cached, "tex", None), _CLOSED_TEXTURE, False):
                self.stats.cache_hits += 1
                return cached
            del self.cache[cache_key]
        if cacheable:
            self.stats.cache_misses += 1
        else:
            self.stats.cache_skips += 1
        return None

    def _render_node(self, node, cacheable, cache_key):
        from .spatial import is_spatial_node
        if is_spatial_node(node):
            return self._render_spatial_node(node, cacheable, cache_key)
        from sdf_ui.gpu.nodes.registry import registry
        plugin = registry.get(node.op)
        if self._can_render_overlay_chain(node):
            result = self._render_overlay_chain(node)
            if result is not None:
                self._store_node_result(node, result, cacheable, cache_key)
                return result
        inputs = tuple(self.render(input_) for input_ in node.inputs)
        params = dict((key, self._resolve(value)) for key, value in node.params)
        scale = getattr(self.ctx, "device_scale", 1.0)
        if scale != 1.0:
            params = _scale_spatial_params(node.op, params, scale)
        result = plugin.render(self, inputs, params)
        self._store_node_result(node, result, cacheable, cache_key)
        self._consume_graph_inputs(node)
        return result

    def _can_render_overlay_chain(self, node):
        return (
            node.op == "alpha_overlay"
            and not self._dirty_regions_enabled
            and self.ctx._active_evaluation_rect is None
        )

    def _render_spatial_node(self, node, cacheable, cache_key):
        from ..gpu.spatial import render_spatial
        result = render_spatial(self, node)
        if cacheable:
            self.cache[cache_key] = result
            self.stats.cache_writes += 1
        elif id(node) != self._root_graph_id:
            self._graph_outputs[id(node)] = result
        return result

    def _store_node_result(self, node, result, cacheable, cache_key):
        if cacheable:
            self.cache[cache_key] = result
            self.stats.cache_writes += 1
        elif (
            not self._dirty_regions_enabled
            and node.op != "output"
            and id(node) != self._root_graph_id
            and isinstance(result, TextureNode)
        ):
            self._graph_outputs[id(node)] = result

    def _consume_graph_inputs(self, node):
        """Recycle uncached outputs once their final graph edge has run."""
        if self._dirty_regions_enabled or getattr(node, "op", None) == "output":
            return
        for child in node.inputs:
            identity = id(child)
            remaining = self._graph_remaining_refs.get(identity)
            if remaining is None:
                continue
            remaining -= 1
            self._graph_remaining_refs[identity] = remaining
            if remaining != 0 or identity == self._root_graph_id:
                continue
            if self._cache_node(child):
                continue
            output = self._graph_outputs.pop(identity, None)
            if output is not None:
                self._close_graph_output(output)

    def _release_unused_graph_outputs(self, root_output):
        """Return uncached outputs from skipped branches to the context image pool."""
        for output in self._graph_outputs.values():
            if output is not root_output:
                self._close_graph_output(output)
        self._graph_outputs.clear()

    @staticmethod
    def _close_graph_output(output):
        outputs = output if isinstance(output, tuple) else (output,)
        for item in outputs:
            if isinstance(item, TextureNode) and item.tex is not None:
                item.close()

    def _eval_dirty_layer(self, node):
        """Reuse per-node outputs and patch pointwise changes into their images."""
        materialized = self._materialized_input(node)
        if materialized is not None:
            return materialized
        from sdf_ui.gpu.nodes.registry import registry
        signature = self._stable_node_key(node)
        state = self._dirty_layer_state(node)
        cached = self._dirty_cache_hit(node, state, signature)
        if cached is not None:
            return cached
        state = self._dirty_layer_state(node)
        return self._render_dirty_node(node, state, signature, registry)

    def _dirty_cache_hit(self, node, state, signature):
        if state is not None:
            if state[0] == signature and not self._layer_output_closed(state[1]):
                self.stats.cache_hits += 1
                self.stats.dirty_layer_hits += 1
                return state[1]
            if state[0] == signature:
                self._forget_dirty_layer(node)
                state = None
        if state is None and node.should_cache:
            cached = self.cache.get(self._node_key(node))
            if cached is not None and not self._layer_output_closed(cached):
                self.stats.cache_hits += 1
                self._remember_dirty_layer(node, signature, cached)
                return cached
            if cached is not None:
                self.cache.pop(self._node_key(node), None)
        return None

    def _render_dirty_node(self, node, state, signature, registry):
        plugin = registry.get(node.op)
        if state is not None or node.should_cache:
            self.stats.cache_misses += 1
        else:
            self.stats.cache_skips += 1

        inputs = tuple(self.render(input_) for input_ in node.inputs)
        params = dict((key, self._resolve(value)) for key, value in node.params)
        scale = getattr(self.ctx, "device_scale", 1.0)
        if scale != 1.0:
            params = _scale_spatial_params(node.op, params, scale)
        region = self.ctx._active_evaluation_rect
        result = self._dirty_plugin_result(node, state, plugin, inputs, params, region, registry)
        self._remember_dirty_layer(node, signature, result)
        if node.should_cache:
            self.cache[signature] = result
            self.stats.cache_writes += 1
        return result

    def _dirty_plugin_result(self, node, state, plugin, inputs, params, region, registry):
        if state is not None and region is not None:
            return self._patch_dirty_region(node, state[1], plugin, inputs, params, region, registry)
        if state is None and region is not None:
            self.stats.dirty_region_fallbacks += 1
            return self._render_full_plugin(plugin, inputs, params)
        return plugin.render(self, inputs, params)

    def _patch_dirty_region(self, node, previous, plugin, inputs, params, region, registry):
        try:
            self._region_children(node, registry)
        except NotImplementedError:
            self.stats.dirty_region_fallbacks += 1
            return self._render_full_plugin(plugin, inputs, params)
        result = plugin.render(self, inputs, params)
        if not self._patch_layer_output(previous, result, region):
            self.stats.dirty_region_fallbacks += 1
            return self._render_full_plugin(plugin, inputs, params)
        self.stats.dirty_layer_updates += 1
        self.stats.dirty_region_pixels += region[2] * region[3]
        return previous

    def _render_full_plugin(self, plugin, inputs, params):
        previous_rect = self.ctx._active_evaluation_rect
        self.ctx._active_evaluation_rect = None
        try:
            return plugin.render(self, inputs, params)
        finally:
            self.ctx._active_evaluation_rect = previous_rect

    def _patch_layer_output(self, destination, source, rect):
        if destination is source:
            return True
        if isinstance(destination, tuple) or isinstance(source, tuple):
            return self._patch_tuple_output(destination, source, rect)
        return self._patch_texture_output(destination, source, rect)

    def _patch_tuple_output(self, destination, source, rect):
        if not isinstance(destination, tuple) or not isinstance(source, tuple):
            return False
        if len(destination) != len(source):
            return False
        return all(self._patch_layer_output(target, update, rect) for target, update in zip(destination, source))

    def _patch_texture_output(self, destination, source, rect):
        if not self._are_patchable_textures(destination, source):
            return False
        dest_tex, source_tex = destination.tex, source.tex
        if dest_tex is source_tex:
            return True
        if dest_tex is None or source_tex is None:
            return False
        x, y, width, height = rect
        if not self._patch_region_matches(dest_tex, source_tex, (x, y, width, height)):
            return False
        self.ctx.copy_image_region(source_tex, dest_tex, rect)
        self._dirty_region_temporaries.append(source_tex)
        return True

    @staticmethod
    def _are_patchable_textures(destination, source):
        return isinstance(destination, TextureNode) and isinstance(source, TextureNode)

    def _patch_region_matches(self, destination, source, rect):
        x, y, width, height = rect
        return (
            tuple(destination.origin) == (0, 0)
            and tuple(destination.size) == tuple(self.ctx.size)
            and tuple(source.origin) == (x, y)
            and tuple(source.size) == (width, height)
            and source.format == destination.format
        )

    @staticmethod
    def _layer_output_closed(output):
        outputs = output if isinstance(output, tuple) else (output,)
        return any(
            getattr(getattr(item, "tex", None), _CLOSED_TEXTURE, False)
            for item in outputs
        )

    def _resolve_dirty_rect(self, root):
        """Find a conservative crop covering changed animation parameters."""
        from ..gpu.regions import normalize_rect

        canvas = (0, 0, *self.ctx.size)
        seed_frame = not self._dirty_layer_cache
        self._dirty_bounds = {}
        self._needs_full_dirty_seed = False
        try:
            rect = self._dirty_bounds_for(root, set(), seed_frame)
        except _FullDirtyFrame:
            rect = canvas
        if self._needs_full_dirty_seed:
            rect = None
        if rect is None:
            return None
        return normalize_rect(rect, self.ctx.size)

    def _dirty_bounds_for(self, node, visited, seed_frame):
        if id(node) in visited:
            return None
        visited.add(id(node))
        if isinstance(node, MultiOutputResult):
            return _union_bounds(
                (self._dirty_bounds_for(child, visited, seed_frame) for child in node.inputs),
                self.ctx.size,
            )
        if not isinstance(node, TextureNode) or node.tex is not None:
            return None
        params = self._dirty_bounds_params(node)
        param_key = self._freeze_runtime(params)
        state = self._dirty_layer_state(node)
        current_key = self._stable_node_key(node)
        children = [self._dirty_bounds_for(child, visited, seed_frame) for child in node.inputs]
        child_bounds = [self._dirty_bounds.get(id(child), (None, None, None))[2] for child in node.inputs]
        current = _node_bounds(node.op, params, node.inputs, self.ctx.size, child_bounds)
        if self._store_first_dirty_bounds(node, state, current_key, param_key, current, seed_frame):
            return current
        changed = self._changed_dirty_bounds(node, state, current_key, param_key, current)
        if changed is not None:
            return changed
        self._validate_dirty_bounds_children(node, children)
        return _union_bounds(children, self.ctx.size)

    def _changed_dirty_bounds(self, node, state, current_key, param_key, current):
        previous_param = state[3] if len(state) > 3 else None
        previous_bounds = state[2]
        self._dirty_bounds[id(node)] = (current_key, param_key, current)
        if state[0] != current_key and previous_param != param_key:
            if current is None or previous_bounds is None:
                raise _FullDirtyFrame
            return _union_bounds((current, previous_bounds), self.ctx.size)
        return None

    def _dirty_bounds_params(self, node):
        params = {key: self._resolve(value) for key, value in node.params}
        scale = getattr(self.ctx, "device_scale", 1.0)
        return _scale_spatial_params(node.op, params, scale) if scale != 1.0 else params

    def _store_first_dirty_bounds(self, node, state, current_key, param_key, current, seed_frame):
        if state is not None:
            return False
        self._dirty_bounds[id(node)] = (current_key, param_key, current)
        self._needs_full_dirty_seed |= seed_frame
        return True

    @staticmethod
    def _validate_dirty_bounds_children(node, children):
        if not any(child is not None for child in children):
            return
        from ..gpu.nodes.registry import registry
        plugin = registry.get(node.op)
        contract = None if plugin.shader is None else plugin.shader.differentiation
        if contract is None or contract.sampling != "pointwise":
            raise _FullDirtyFrame

    def _dirty_layer_state(self, node):
        entry = self._dirty_layer_cache.get(id(node))
        if entry is None or entry[0]() is not node:
            return None
        return entry[1]

    def _remember_dirty_layer(self, node, signature, output):
        identity = id(node)
        renderer_ref = weakref.ref(self)

        def remove(node_ref):
            renderer = renderer_ref()
            if renderer is not None:
                current = renderer._dirty_layer_cache.get(identity)
                if current is not None and current[0] is node_ref:
                    renderer._dirty_layer_cache.pop(identity, None)

        info = getattr(self, "_dirty_bounds", {}).get(identity)
        param_key = None if info is None else info[1]
        bounds = None if info is None else info[2]
        self._dirty_layer_cache[identity] = (
            weakref.ref(node, remove),
            (signature, output, bounds, param_key),
        )

    def _forget_dirty_layer(self, node):
        self._dirty_layer_cache.pop(id(node), None)

    def _stable_node_key(self, node):
        previous_rect = self.ctx._active_evaluation_rect
        self.ctx._active_evaluation_rect = None
        try:
            return self._node_key(node)
        finally:
            self.ctx._active_evaluation_rect = previous_rect

    def _render_overlay_chain(self, root):
        from sdf_ui.gpu.nodes.layer.alpha_overlay.plugin import (
            MAX_BATCH_OVERLAYS,
            render_alpha_overlay_batch,
        )

        chain_info = self._overlay_chain(root)
        if chain_info is None:
            return None
        base, chain = chain_info
        current = self.render(base)
        for start in range(0, len(chain), MAX_BATCH_OVERLAYS):
            group = chain[start : start + MAX_BATCH_OVERLAYS]
            current = self._render_overlay_group(
                root, group, current, render_alpha_overlay_batch
            )
        return current

    def _overlay_chain(self, root):
        outer_to_inner = [root]
        base = root.inputs[0]
        while (
            isinstance(base, TextureNode)
            and base.tex is None
            and base.op == "alpha_overlay"
        ):
            base_key = self._node_key(base)
            cached = self.cache.get(base_key) if base.should_cache else None
            if cached is not None:
                if not getattr(getattr(cached, "tex", None), _CLOSED_TEXTURE, False):
                    break
                self.cache.pop(base_key, None)
            outer_to_inner.append(base)
            base = base.inputs[0]

        if len(outer_to_inner) < 2:
            return None
        chain = tuple(reversed(outer_to_inner))
        chain_ids = {id(item) for item in chain}
        if any(
            self._references_overlay_chain(item.inputs[1], chain_ids)
            for item in chain
        ):
            return None
        return base, chain

    def _render_overlay_group(self, root, group, current, render_batch):
        tops = [self._overlay_top(root, item) for item in group]
        output_indices = self._overlay_output_indices(root, group)
        outputs = render_batch(self, current, tops, output_indices=output_indices)
        self._cache_overlay_outputs(root, group, outputs)
        current = outputs[len(group) - 1]
        last_item = group[-1]
        if last_item is not root and not last_item.should_cache:
            self._graph_outputs[id(last_item)] = current
        for item in group:
            self._consume_graph_inputs(item)
        return current

    def _overlay_top(self, root, item):
        if item is not root:
            if item.should_cache:
                self.stats.cache_misses += 1
            else:
                self.stats.cache_skips += 1
        return self.render(item.inputs[1])

    def _overlay_output_indices(self, root, group):
        indices = []
        for index, item in enumerate(group):
            is_last = index == len(group) - 1
            is_shared_cache = item is not root and item.should_cache and (
                item._cache_boundary or self._graph_ref_counts.get(id(item), 0) > 1
            )
            if is_last or is_shared_cache:
                indices.append(index)
        return tuple(indices)

    def _cache_overlay_outputs(self, root, group, outputs):
        for index, item in enumerate(group):
            output = outputs.get(index)
            if output is None or item is root or not item.should_cache:
                continue
            self.cache[self._node_key(item)] = output
            self.stats.cache_writes += 1

    @staticmethod
    def _references_overlay_chain(value, chain_ids):
        pending = [value]
        visited = set()
        while pending:
            item = pending.pop()
            if id(item) in visited:
                continue
            visited.add(id(item))
            if id(item) in chain_ids:
                return True
            if isinstance(item, MultiOutputResult):
                pending.extend(item.inputs)
            elif isinstance(item, TextureNode) and item.tex is None:
                pending.extend(item.inputs)
        return False


    def _resolve(self, value):
        if isinstance(value, TextureNode) or isinstance(value, MultiOutputResult):
            return self._eval(value)
        from ..learning.autodiff import Tensor
        from .expressions import Expr, evaluate_expr

        if isinstance(value, (Expr, Tensor, str)):
            value = evaluate_expr(value, self, self.params)
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
                getattr(node.tex, "generation", None),
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
            (key, self._parameter_key(value)) for key, value in node.params
        )
        return context_key, node.kind, node.op, input_keys, param_keys

    def _parameter_key(self, value):
        from .spatial import is_spatial

        if not is_spatial(value):
            return self._freeze_runtime(self._resolve(value))
        from .expressions import Expr

        if isinstance(value, Expr):
            if value.op == "texture_sample":
                node, index, scope = value.args
                return value.op, self._node_key(node), index, scope
            return value.op, tuple(self._parameter_key(arg) for arg in value.args)
        return tuple(self._parameter_key(arg) for arg in value)

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

    def channel(self, name: str | int = 0):
        """Select a numeric channel for a spatial parameter expression."""
        from .spatial import channel

        return channel(self, name)

    def repeat_grid(
        self,
        columns,
        rows,
        spacing,
        center,
        *,
        stamp_center=None,
        rotation=0.0,
        scale=1.0,
        baked: bool = False,
    ):
        """Repeat a scene with optional spatial bindings and per-copy transforms.

        Procedural templates default to a stamp centered at (0, 0) and
        are clipped to their cells. Set ``stamp_center`` for another origin.
        Ordinary SDF repeats without additional options keep their existing
        behavior. Use ``baked=True`` to repeat an already rendered design.
        """
        from numbers import Integral, Real

        from .spatial import is_spatial, spatial_scene

        if self._can_use_legacy_grid_repeat(
            spacing, center, stamp_center, rotation, scale, baked, Real, is_spatial, spatial_scene
        ):
            return build("repeat_grid", self, columns, rows, spacing, center)
        self._validate_repeat_counts(columns, rows, Integral)
        return self._spatial_repeat(
            columns, rows, spacing, center, stamp_center, rotation, scale, baked
        )

    def _can_use_legacy_grid_repeat(
        self, spacing, center, stamp_center, rotation, scale, baked, Real, is_spatial, spatial_scene
    ):
        return (
            self.kind == "sdf" and self.tex is None and stamp_center is None
            and isinstance(rotation, Real) and rotation == 0.0
            and isinstance(scale, Real) and scale == 1.0 and not baked
            and not spatial_scene(self) and not is_spatial((spacing, center))
        )

    @staticmethod
    def _validate_repeat_counts(columns, rows, Integral):
        if all(not isinstance(count, bool) and isinstance(count, Integral) and count > 0 for count in (columns, rows)):
            return
        raise ValueError("Spatial repeat rows and columns must be fixed positive integers")

    def repeat(
        self,
        s=15.0,
        *,
        center=None,
        stamp_center=None,
        rotation=0.0,
        scale=1.0,
        baked: bool = False,
    ):
        """Repeat a stamp across the canvas with scalar or XY spacing."""
        from numbers import Real

        from .spatial import spatial_scene

        if self._can_use_legacy_repeat(s, center, stamp_center, rotation, scale, baked, Real, spatial_scene):
            return build("repeat", self, s)
        return self._spatial_repeat(
            None,
            None,
            s,
            (0, 0) if center is None else center,
            stamp_center,
            rotation,
            scale,
            baked,
        )

    def _can_use_legacy_repeat(self, spacing, center, stamp_center, rotation, scale, baked, Real, spatial_scene):
        if not self._has_legacy_repeat_defaults(
            spacing, center, stamp_center, rotation, scale, baked, Real, spatial_scene
        ):
            return False
        from .spatial import is_spatial
        return not is_spatial(spacing)

    def _has_legacy_repeat_defaults(
        self, spacing, center, stamp_center, rotation, scale, baked, Real, spatial_scene
    ):
        return all((
            self.kind == "sdf", self.tex is None, center is None,
            stamp_center is None, isinstance(rotation, Real), rotation == 0,
            isinstance(scale, Real), scale == 1, not baked,
            not isinstance(spacing, (tuple, list)), not spatial_scene(self),
        ))

    def _spatial_repeat(
        self, columns, rows, spacing, center, stamp_center, rotation, scale, baked
    ):
        if not isinstance(spacing, (tuple, list)):
            spacing = (spacing, spacing)
        if stamp_center is None:
            if self.tex is not None:
                device_scale = getattr(self.context, "device_scale", 1.0)
                stamp_center = tuple(
                    (offset + size * 0.5) / device_scale
                    for offset, size in zip(self.origin, self.size)
                )
            else:
                stamp_center = center if baked else (0.0, 0.0)
        kwargs = dict(mode=self.mode) if self.kind == "color" else {}
        return type(self)(
            op="spatial_repeat",
            inputs=(self,),
            params=dict(
                columns=columns,
                rows=rows,
                spacing=spacing,
                center=center,
                stamp_center=stamp_center,
                rotation=rotation,
                scale=scale,
                baked=baked,
            ),
            **kwargs,
        )

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
        cache_boundary=False,
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
        self._cache_boundary = cache_boundary or label is not None

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
        if self.tex is not None and getattr(self.tex, _CLOSED_TEXTURE, False):
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

    def render(
        self,
        ctx=None,
        params=None,
        cache=None,
        evaluation_rect=None,
        *,
        progress=0.0,
        canvas=None,
    ) -> Self:
        """Materialize this node, optionally cropping to a canvas pixel rect.

        The rectangle is ``(x, y, width, height)``. All shaders in the render
        tree use the same region, so this currently supports pointwise
        operations and rejects filters that need neighboring pixels.
        """
        ctx = _resolve_canvas_alias(ctx, canvas)
        if self.tex is not None and getattr(self.tex, _CLOSED_TEXTURE, False):
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
        ).render(self, progress=progress)

    def read(self) -> NDArray[Any]:
        """Copy this materialized texture's pixels into a NumPy array.

        SDF textures return a two-dimensional float array. Color textures
        return an ``(height, width, 4)`` array in the image's native dtype.
        """
        if self.tex is None:
            raise RuntimeError(
                "Texture node must be rendered before its pixels can be read"
            )
        if getattr(self.tex, _CLOSED_TEXTURE, False):
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
        if getattr(self.tex, _CLOSED_TEXTURE, False):
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
        progress: float = 0.0,
        *,
        canvas: Any = None,
    ) -> None:
        ctx = _resolve_canvas_alias(ctx, canvas)

        def _show(texture):
            converted = None
            if (
                getattr(texture, "mode", None) == "LINEAR"
                or (getattr(texture, "mode", None) == "LAB" and conversion)
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
            progress=progress,
        )

    def save(
        self,
        name: str | Path,
        ctx: Any = None,
        params: dict[str, Any] | None = None,
        cache: dict[Any, Any] | None = None,
        conversion: bool = True,
        size: tuple[int, int] | None = None,
        evaluation_rect: tuple[int, int, int, int] | None = None,
        progress: float = 0.0,
        *,
        canvas: Any = None,
    ) -> Path:
        ctx = _resolve_canvas_alias(ctx, canvas)
        name = Path(name)

        def _save(texture):
            converted = None
            if self.kind == "color":
                if (
                    getattr(texture, "mode", None) == "LINEAR"
                    or (getattr(texture, "mode", None) == "LAB" and conversion)
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
            progress=progress,
        )
        return name

    def named(self, label: str):
        return self._clone(label=label, should_cache=self.should_cache)

    def cache(self, label: Optional[str] = None):
        return self._clone(
            label=label or self.label, should_cache=True, cache_boundary=True
        )

    def uncached(self):
        """Skip cross-render caching and permit reuse after the final graph use."""
        return self._clone(label=self.label, should_cache=False)

    def _render_for_io(self, ctx, params, cache, size, evaluation_rect, progress):
        return self.render(
            ctx,
            params=params,
            cache=cache,
            evaluation_rect=evaluation_rect,
            progress=progress,
        )

    def _with_rendered_texture(
        self, func, *, ctx, params, cache, size, evaluation_rect, progress
    ):
        if ctx is not None:
            texture = self._render_for_io(
                ctx, params, cache, size, evaluation_rect, progress
            )
            try:
                return func(texture)
            finally:
                if texture is not self and self.tex is None:
                    texture.close()

        if size is not None:
            with Context(size) as temp_ctx:
                return func(
                    self._render_for_io(
                        temp_ctx, params, cache, size, evaluation_rect, progress
                    )
                )

        if self.context is None:
            raise ValueError(
                "A Context is required to save or show an unmaterialized texture; "
                "pass ctx= or size= to create a temporary context"
            )
        return func(
            self._render_for_io(
                self.context, params, cache, size, evaluation_rect, progress
            )
        )

    def _clone(self, *, label, should_cache, cache_boundary=None):
        kwargs = {}
        if hasattr(self, "mode"):
            kwargs["mode"] = self.mode
        return type(self)(
            op=self.op,
            inputs=self.inputs,
            params=self.params,
            label=label,
            should_cache=should_cache,
            cache_boundary=(
                self._cache_boundary if cache_boundary is None else cache_boundary
            ),
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


class WarpableTextureNode(TextureNode):
    """Texture node variant that supports vector-field displacement."""

    _warp_operation = None

    def warp(self, field, strength=1.0, steps=1):
        if self._warp_operation is None:
            raise AttributeError(f"{type(self).__name__} does not support warp")
        from ._texture_ops import warp_texture

        return warp_texture(self._warp_operation, self, field, strength, steps)


@dataclass(frozen=True)
class MultiOutputResult:
    op: str
    inputs: tuple = ()
    params: tuple = ()
    label: Optional[str] = None
    should_cache: bool = True
    kind: str = "multi_output"

    def render(
        self, ctx, params=None, cache=None, evaluation_rect=None, *, progress=0.0
    ):
        return Renderer(
            ctx,
            params=params,
            cache=cache,
            evaluation_rect=evaluation_rect,
        ).render(self, progress=progress)

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
