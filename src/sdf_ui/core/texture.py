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
    x1, y1 = min(width, int(x1 + 0.999) + 2), min(height, int(y1 + 0.999) + 2)
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1 - x0, y1 - y0)


def _node_bounds(op, params, inputs, canvas_size, child_bounds=()):
    """Return conservative pixel bounds for common finite SDF primitives."""
    center = params.get("center", params.get("offset"))
    if op in {"circle", "ring", "ellipse", "rect", "rounded_rect", "ngon"}:
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
        angle = abs(float(params.get("angle", 0)))
        if angle:
            import math

            diagonal = math.hypot(rx, ry)
            rx = ry = diagonal
        return _union_bounds(((x - rx, y - ry, 2 * rx, 2 * ry),), canvas_size)
    if op in {"capsule", "segment"}:
        a, b = params.get("a"), params.get("b")
        if a is None or b is None:
            return None
        radius = abs(float(params.get("radius", params.get("thickness", 0))))
        return _union_bounds(((min(a[0], b[0]) - radius, min(a[1], b[1]) - radius,
                               abs(a[0] - b[0]) + 2 * radius,
                               abs(a[1] - b[1]) + 2 * radius),), canvas_size)
    if op in {"fill", "outline"} and child_bounds and child_bounds[0] is not None:
        x, y, width, height = child_bounds[0]
        margin = max(
            abs(float(params.get("inflate", 0) or 0)),
            abs(float(params.get("inner", 0) or 0)),
            abs(float(params.get("outer", 0) or 0)),
        ) + 2
        return _union_bounds(
            ((x - margin, y - margin, width + 2 * margin, height + 2 * margin),),
            canvas_size,
        )
    if op in {
        "union", "smooth_union", "intersection", "subtract", "smooth_subtract",
        "smooth_intersection", "alpha_overlay", "over",
        "isolines", "transparency", "layer_mask", "to_rgb", "to_lab",
    }:
        selected = child_bounds[:1] if op in {"isolines", "transparency", "to_rgb", "to_lab"} else child_bounds
        return _union_bounds(selected, canvas_size)
    if op in {"translate", "rotate", "scale"} and child_bounds:
        child = child_bounds[0]
        if child is None:
            return None
        x, y, width, height = child
        cx, cy = params.get("center", (0.0, 0.0))
        points = ((x, y), (x + width, y), (x, y + height), (x + width, y + height))
        import math

        transformed = []
        for px, py in points:
            if op == "translate":
                ox, oy = params.get("offset", (0.0, 0.0))
                px, py = px + ox, py + oy
            elif op == "scale":
                factor = params.get("factor", 1.0)
                px, py = cx + (px - cx) * factor, cy + (py - cy) * factor
            else:
                angle = params.get("angle", 0.0)
                dx, dy = px - cx, py - cy
                px, py = cx + dx * math.cos(angle) - dy * math.sin(angle), cy + dx * math.sin(angle) + dy * math.cos(angle)
            transformed.append((px, py))
        xs, ys = zip(*transformed)
        return _union_bounds(((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),), canvas_size)
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
    for key in _SPATIAL_PARAMS.intersection(result):
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
    region, and ``stats`` and ``cache_info()`` expose work performed.
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
        if is_root_render:
            from ..gpu.regions import normalize_rect
            from .expressions import validate_progress

            if progress is not None:
                self.progress = validate_progress(progress)
            if dirty_rect is not None and self.evaluation_rect is not None:
                raise ValueError(
                    "dirty_rect cannot be combined with evaluation_rect"
                )
            auto_dirty = dirty_rect == "auto" or (
                dirty_rect is None and self._dirty_regions_enabled
            )
            self._dirty_regions_enabled |= dirty_rect is not None
            if auto_dirty:
                rect = self._resolve_dirty_rect(value)
            else:
                rect = (
                    normalize_rect(dirty_rect, self.ctx.size)
                    if dirty_rect is not None
                    else self._resolve_evaluation_rect(value)
                )
            self._graph_ref_counts = self._count_graph_references(value)
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
            if self.evaluation_rect is not None:
                self._validate_region_graph(value)
            self.ctx._active_evaluation_rect = rect
            self.ctx._active_render_stats = self.stats
            self.stats.gpu_timing_supported = self.ctx.gpu_timing_supported
            self.stats.render_calls += 1
            self.ctx._begin_command_batch()

        self._render_depth += 1
        try:
            if isinstance(value, TextureNode) or isinstance(value, MultiOutputResult):
                if is_root_render and self._dirty_regions_enabled:
                    return self._eval_dirty_layer(value)
                return self._eval(value)
            return self._resolve(value)
        finally:
            self._render_depth -= 1
            if is_root_render:
                self.ctx._end_command_batch()
                for temporary in self._dirty_region_temporaries:
                    temporary.close()
                self._dirty_region_temporaries.clear()
                self.ctx._resolve_shader_timings(self.stats)
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

    @staticmethod
    def _count_graph_references(root):
        counts = {}
        visited = set()
        pending = [root]
        while pending:
            value = pending.pop()
            if not isinstance(value, (TextureNode, MultiOutputResult)):
                continue
            if isinstance(value, TextureNode) and value.tex is not None:
                continue
            for child in value.inputs:
                if isinstance(child, (TextureNode, MultiOutputResult)):
                    counts[id(child)] = counts.get(id(child), 0) + 1
                    if id(child) not in visited:
                        pending.append(child)
            visited.add(id(value))
        return counts

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
        if isinstance(value, TextureNode) and value.tex is not None:
            return ()
        if not isinstance(value, (TextureNode, MultiOutputResult)):
            return ()
        if isinstance(value, TextureNode) and value.op == "output":
            return value.inputs
        plugin = registry.get(value.op)
        shader = plugin.shader
        if value.op == "output":
            return value.inputs
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
        self._dirty_layer_cache.clear()
        self._dirty_region_temporaries.clear()

    def clear_cache(self):
        self.cache.clear()
        self._dirty_layer_cache.clear()
        self._dirty_region_temporaries.clear()

    def cache_info(self):
        return RenderCacheInfo(
            entries=len(self.cache),
            hits=self.stats.cache_hits,
            misses=self.stats.cache_misses,
            writes=self.stats.cache_writes,
            skips=self.stats.cache_skips,
        )

    def _eval(self, node):
        if self._dirty_regions_enabled:
            return self._eval_dirty_layer(node)
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
        if node.op == "alpha_overlay" and not self._dirty_regions_enabled:
            result = self._render_overlay_chain(node)
            if result is not None:
                if node.should_cache:
                    self.cache[cache_key] = result
                    self.stats.cache_writes += 1
                return result

        inputs = tuple(self.render(input_) for input_ in node.inputs)
        params = dict((key, self._resolve(value)) for key, value in node.params)
        scale = getattr(self.ctx, "device_scale", 1.0)
        if scale != 1.0:
            params = _scale_spatial_params(node.op, params, scale)
        result = plugin.render(self, inputs, params)

        if node.should_cache:
            self.cache[cache_key] = result
            self.stats.cache_writes += 1

        return result

    def _eval_dirty_layer(self, node):
        """Reuse per-node outputs and patch pointwise changes into their images."""
        if isinstance(node, TextureNode) and getattr(node, "tex", None) is not None:
            if getattr(node.tex, "closed", False):
                raise RuntimeError("Cannot use a closed texture node as an input")
            return node

        from sdf_ui.gpu.nodes.registry import registry

        signature = self._stable_node_key(node)
        state = self._dirty_layer_state(node)
        if state is not None and state[0] == signature:
            if not self._layer_output_closed(state[1]):
                self.stats.cache_hits += 1
                self.stats.dirty_layer_hits += 1
                return state[1]
            self._forget_dirty_layer(node)
            state = None

        cache_key = self._node_key(node)
        if state is None and node.should_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            if not self._layer_output_closed(cached):
                self.stats.cache_hits += 1
                self._remember_dirty_layer(node, signature, cached)
                return cached
            self.cache.pop(cache_key, None)

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
        if state is not None and region is not None:
            try:
                self._region_children(node, registry)
            except NotImplementedError:
                result = self._render_full_plugin(plugin, inputs, params)
                self.stats.dirty_region_fallbacks += 1
            else:
                result = plugin.render(self, inputs, params)
                if self._patch_layer_output(state[1], result, region):
                    result = state[1]
                    self.stats.dirty_layer_updates += 1
                    self.stats.dirty_region_pixels += region[2] * region[3]
                else:
                    result = self._render_full_plugin(plugin, inputs, params)
                    self.stats.dirty_region_fallbacks += 1
        elif state is None and region is not None:
            result = self._render_full_plugin(plugin, inputs, params)
            self.stats.dirty_region_fallbacks += 1
        else:
            result = plugin.render(self, inputs, params)

        self._remember_dirty_layer(node, signature, result)
        if node.should_cache:
            self.cache[signature] = result
            self.stats.cache_writes += 1
        return result

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
        if isinstance(destination, tuple) and isinstance(source, tuple):
            if len(destination) != len(source):
                return False
            return all(
                self._patch_layer_output(target, update, rect)
                for target, update in zip(destination, source)
            )
        if not isinstance(destination, TextureNode) or not isinstance(
            source, TextureNode
        ):
            return False
        dest_tex, source_tex = destination.tex, source.tex
        if dest_tex is source_tex:
            return True
        if dest_tex is None or source_tex is None:
            return False
        x, y, width, height = rect
        if (
            tuple(dest_tex.origin) != (0, 0)
            or tuple(dest_tex.size) != tuple(self.ctx.size)
            or tuple(source_tex.origin) != (x, y)
            or tuple(source_tex.size) != (width, height)
            or source_tex.format != dest_tex.format
        ):
            return False
        self.ctx.copy_image_region(source_tex, dest_tex, rect)
        self._dirty_region_temporaries.append(source_tex)
        return True

    @staticmethod
    def _layer_output_closed(output):
        outputs = output if isinstance(output, tuple) else (output,)
        return any(
            getattr(getattr(item, "tex", None), "closed", False)
            for item in outputs
        )

    def _resolve_dirty_rect(self, root):
        """Find a conservative crop covering changed animation parameters."""
        from ..gpu.regions import normalize_rect

        canvas = (0, 0, *self.ctx.size)
        seed_frame = not self._dirty_layer_cache
        visited = set()

        def bounds(node):
            if id(node) in visited:
                return None
            visited.add(id(node))
            if isinstance(node, MultiOutputResult):
                children = [bounds(child) for child in node.inputs]
                return _union_bounds(children, self.ctx.size)
            if not isinstance(node, TextureNode) or node.tex is not None:
                return None
            params = {key: self._resolve(value) for key, value in node.params}
            scale = getattr(self.ctx, "device_scale", 1.0)
            if scale != 1.0:
                params = _scale_spatial_params(node.op, params, scale)
            param_key = self._freeze_runtime(params)
            state = self._dirty_layer_state(node)
            current_key = self._stable_node_key(node)
            children = [bounds(child) for child in node.inputs]
            child_bounds = [
                self._dirty_bounds.get(id(child), (None, None, None))[2]
                for child in node.inputs
            ]
            current_bounds = _node_bounds(
                node.op, params, node.inputs, self.ctx.size, child_bounds
            )
            if state is None:
                self._dirty_bounds[id(node)] = (current_key, param_key, current_bounds)
                self._needs_full_dirty_seed |= seed_frame
                return current_bounds
            previous_param_key = state[3] if len(state) > 3 else None
            previous_bounds = state[2]
            if state[0] != current_key and previous_param_key != param_key:
                if current_bounds is None or previous_bounds is None:
                    raise _FullDirtyFrame
                changed = _union_bounds((current_bounds, previous_bounds), self.ctx.size)
                self._dirty_bounds[id(node)] = (current_key, param_key, current_bounds)
                return changed
            self._dirty_bounds[id(node)] = (current_key, param_key, current_bounds)
            if any(child is not None for child in children):
                from ..gpu.nodes.registry import registry

                plugin = registry.get(node.op)
                contract = None if plugin.shader is None else plugin.shader.differentiation
                if contract is None or contract.sampling != "pointwise":
                    raise _FullDirtyFrame
            return _union_bounds(children, self.ctx.size)

        self._dirty_bounds = {}
        self._needs_full_dirty_seed = False
        try:
            rect = bounds(root)
        except _FullDirtyFrame:
            rect = canvas
        if self._needs_full_dirty_seed:
            rect = None
        if rect is None:
            return None
        return normalize_rect(rect, self.ctx.size)

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
                if not getattr(getattr(cached, "tex", None), "closed", False):
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

        current = self.render(base)
        for start in range(0, len(chain), MAX_BATCH_OVERLAYS):
            group = chain[start : start + MAX_BATCH_OVERLAYS]
            tops = []
            for item in group:
                if item is not root:
                    if item.should_cache:
                        self.stats.cache_misses += 1
                    else:
                        self.stats.cache_skips += 1
                tops.append(self.render(item.inputs[1]))

            output_indices = tuple(
                index
                for index, item in enumerate(group)
                if index == len(group) - 1
                or (
                    item is not root
                    and item.should_cache
                    and (
                        item._cache_boundary
                        or self._graph_ref_counts.get(id(item), 0) > 1
                    )
                )
            )
            outputs = render_alpha_overlay_batch(
                self, current, tops, output_indices=output_indices
            )
            for index, item in enumerate(group):
                output = outputs.get(index)
                if output is not None and item is not root and item.should_cache:
                    self.cache[self._node_key(item)] = output
                    self.stats.cache_writes += 1
            current = outputs[len(group) - 1]
        return current

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

    def render(
        self, ctx=None, params=None, cache=None, evaluation_rect=None, *, progress=0.0
    ) -> Self:
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
        progress: float = 0.0,
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
            progress=progress,
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
        progress: float = 0.0,
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
            progress=progress,
        )

    def named(self, label: str):
        return self._clone(label=label, should_cache=self.should_cache)

    def cache(self, label: Optional[str] = None):
        return self._clone(
            label=label or self.label, should_cache=True, cache_boundary=True
        )

    def uncached(self):
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
