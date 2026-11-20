from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

_COLORNAMESPACE_TEXT = "ColorNamespace"

_COLORTEXTURE_TEXT = "ColorTexture"

_FIELDNAMESPACE_TEXT = "FieldNamespace"

_FIELDTEXTURE_TEXT = "FieldTexture"

_POSTNAMESPACE_TEXT = "PostNamespace"

_RENDERER_TEXT = "Renderer"

_SDFNAMESPACE_TEXT = "SDFNamespace"

_SDFTEXTURE_TEXT = "SDFTexture"

_TEXTURENODE_TEXT = "TextureNode"

_FROM___FUTURE___IMPORT_ANNOTATIONS_TEXT = "from __future__ import annotations"

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PACKAGE = SRC / "sdf_ui"
CORE = PACKAGE / "core"


def _ensure_src_on_path() -> None:
    src_path = str(SRC)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _ensure_root_on_path() -> None:
    root_path = str(ROOT)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)


def _load_runtime_objects():
    _ensure_src_on_path()

    from sdf_ui.core.api import ColorNamespace, SDFNamespace
    from sdf_ui.core.color import ColorTexture
    from sdf_ui.core.field import FieldNamespace, FieldTexture
    from sdf_ui.core.sdf import SDFTexture
    from sdf_ui.core.texture import (
        PostNamespace,
        RenderCacheInfo,
        Renderer,
        RenderStats,
        TextureNode,
    )
    from sdf_ui.gpu.nodes.base import TextureKind
    from sdf_ui.gpu.nodes.loader import load_plugins
    from sdf_ui.gpu.nodes.registry import PluginRegistry

    registry = PluginRegistry()
    # Distributed stubs describe builtins, independent of third-party packages
    # installed in the developer's environment.
    load_plugins(registry)
    registry._loaded = True

    return {
        _COLORNAMESPACE_TEXT: ColorNamespace,
        _FIELDNAMESPACE_TEXT: FieldNamespace,
        _FIELDTEXTURE_TEXT: FieldTexture,
        _SDFNAMESPACE_TEXT: SDFNamespace,
        _COLORTEXTURE_TEXT: ColorTexture,
        _SDFTEXTURE_TEXT: SDFTexture,
        _POSTNAMESPACE_TEXT: PostNamespace,
        _RENDERER_TEXT: Renderer,
        "RenderStats": RenderStats,
        "RenderCacheInfo": RenderCacheInfo,
        _TEXTURENODE_TEXT: TextureNode,
        "TextureKind": TextureKind,
        "registry": registry,
    }


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _format_default(value: Any) -> str:
    return repr(value)


def _format_annotation(annotation: Any) -> str:
    if annotation is inspect._empty:
        return "Any"
    if annotation is None:
        return "None"
    if annotation is type(None):
        return "None"
    if isinstance(annotation, str):
        return annotation
    origin = get_origin(annotation)
    if origin in {Union, UnionType}:
        return " | ".join(_format_annotation(arg) for arg in get_args(annotation))
    if origin is not None:
        origin_name = getattr(origin, "__name__", str(origin))
        args = ", ".join(_format_annotation(arg) for arg in get_args(annotation))
        return f"{origin_name}[{args}]"
    return getattr(annotation, "__name__", str(annotation))


def _format_manual_method(name: str, func: Any, return_type: str) -> str:
    signature = inspect.signature(func)
    rendered = []
    has_star = False
    for parameter in signature.parameters.values():
        chunk = parameter.name
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            chunk = f"*{parameter.name}: {_format_annotation(parameter.annotation)}"
            has_star = True
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            chunk = f"**{parameter.name}: {_format_annotation(parameter.annotation)}"
        elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            if not has_star:
                rendered.append("*")
                has_star = True
            chunk = f"{parameter.name}: {_format_annotation(parameter.annotation)}"
        elif parameter.name != "self":
            chunk = f"{parameter.name}: {_format_annotation(parameter.annotation)}"

        if parameter.default is not inspect._empty and parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            chunk = f"{chunk} = {_format_default(parameter.default)}"
        rendered.append(chunk)

    return f"    def {name}({', '.join(rendered)}) -> {return_type}: ..."


def _format_plugin_method(plugin: Any, return_type: str) -> str:
    rendered = ["self"]
    rendered.extend(f"{name}: Any" for name in _plugin_input_arg_names(plugin))
    defaults = plugin.defaults or {}
    for param in plugin.params:
        chunk = f"{param}: Any"
        if param in defaults:
            chunk = f"{chunk} = {_format_default(defaults[param])}"
        rendered.append(chunk)
    return f"    def {plugin.name}({', '.join(rendered)}) -> {return_type}: ..."


def _plugin_input_arg_names(plugin: Any) -> tuple[str, ...]:
    return plugin.method_input_names()


class _StubRenderer:
    @staticmethod
    def _render_class(
        name: str, body_lines: list[str], *, properties: list[str] | None = None
    ) -> str:
        members = [*(properties or []), *body_lines]
        body = "\n".join(members) if members else "    pass"
        return f"class {name}:\n{body}\n"

    @staticmethod
    def _plugin_return_type(texture_kind: Any, plugin: Any) -> str:
        if plugin.result == texture_kind.SDF:
            return _SDFTEXTURE_TEXT
        if plugin.result == texture_kind.COLOR:
            return _COLORTEXTURE_TEXT
        return "MultiOutputResult"

    @staticmethod
    def _method_return_overrides() -> dict[str, dict[str, str]]:
        return {
            _TEXTURENODE_TEXT: {
                "close": "None",
                "__enter__": "Self",
                "__exit__": "None",
                "render": "Self",
                "read": "NDArray[Any]",
                "show": "None",
                "save": "Path",
                "channel": "TextureChannel",
                "repeat": "Self",
                "repeat_grid": "Self",
                "named": "Any",
                "cache": "Any",
                "uncached": "Any",
            },
            _RENDERER_TEXT: {
                "__init__": "None",
                "render": "Any",
                "close": "None",
                "cache_info": "RenderCacheInfo",
                "__enter__": _RENDERER_TEXT,
                "__exit__": "None",
            },
            _SDFTEXTURE_TEXT: {
                "fill": _COLORTEXTURE_TEXT,
                "ink": _COLORTEXTURE_TEXT,
                "stroke": _COLORTEXTURE_TEXT,
                "ink_linear": _COLORTEXTURE_TEXT,
                "fill_linear": _COLORTEXTURE_TEXT,
                "__or__": _SDFTEXTURE_TEXT,
                "__and__": _SDFTEXTURE_TEXT,
                "__sub__": _SDFTEXTURE_TEXT,
                "__xor__": _SDFTEXTURE_TEXT,
                "xor": _SDFTEXTURE_TEXT,
                "intersection": _SDFTEXTURE_TEXT,
                "intersect": _SDFTEXTURE_TEXT,
                "mask": _COLORTEXTURE_TEXT,
                "debug_seams": _COLORTEXTURE_TEXT,
                "debug_distance": _COLORTEXTURE_TEXT,
                "debug_boolean": _COLORTEXTURE_TEXT,
                "repeat_grid": _SDFTEXTURE_TEXT,
                "warp": _SDFTEXTURE_TEXT,
                "domain_warp": _SDFTEXTURE_TEXT,
                "turbulence": _SDFTEXTURE_TEXT,
            },
            _SDFNAMESPACE_TEXT: {
                "union_all": _SDFTEXTURE_TEXT,
                "rect_from_size": _SDFTEXTURE_TEXT,
                "rounded_rect_from_size": _SDFTEXTURE_TEXT,
                "star": _SDFTEXTURE_TEXT,
                "fir_tree": _SDFTEXTURE_TEXT,
                "isometric_cube": _SDFTEXTURE_TEXT,
                "arrow": _SDFTEXTURE_TEXT,
                "polyline": _SDFTEXTURE_TEXT,
                "svg": _SDFTEXTURE_TEXT,
                "math": _SDFTEXTURE_TEXT,
            },
            _COLORNAMESPACE_TEXT: {
                "linear_rgba": _COLORTEXTURE_TEXT,
                "concentric_rings": _COLORTEXTURE_TEXT,
                "palette": "Palette",
            },
            _FIELDNAMESPACE_TEXT: {
                "constant": _FIELDTEXTURE_TEXT,
                "radial": _FIELDTEXTURE_TEXT,
                "vortex": _FIELDTEXTURE_TEXT,
                "turbulence": _FIELDTEXTURE_TEXT,
            },
            _FIELDTEXTURE_TEXT: {
                "scale": _FIELDTEXTURE_TEXT,
                "add": _FIELDTEXTURE_TEXT,
                "visualize": _COLORTEXTURE_TEXT,
                "contours": _COLORTEXTURE_TEXT,
            },
            _COLORTEXTURE_TEXT: {
                "warp": _COLORTEXTURE_TEXT,
                "over": _COLORTEXTURE_TEXT,
                "alpha_overlay": _COLORTEXTURE_TEXT,
                "to_linear": _COLORTEXTURE_TEXT,
                "expose": _COLORTEXTURE_TEXT,
                "film_grain": _COLORTEXTURE_TEXT,
                "bloom": _COLORTEXTURE_TEXT,
                "finish": _COLORTEXTURE_TEXT,
                "to_lab": _COLORTEXTURE_TEXT,
                "to_rgb": _COLORTEXTURE_TEXT,
            },
            _POSTNAMESPACE_TEXT: {
                "to_lab": _COLORTEXTURE_TEXT,
                "to_rgb": _COLORTEXTURE_TEXT,
            },
        }

    @staticmethod
    def _collect_manual_methods(class_name: str, cls: Any) -> list[str]:
        overrides = _method_return_overrides().get(class_name, {})
        methods = []
        for name, value in cls.__dict__.items():
            if (
                name.startswith("_")
                and name
                not in {
                    "__enter__",
                    "__exit__",
                    "__or__",
                    "__and__",
                    "__sub__",
                    "__xor__",
                }
                and not (class_name == _RENDERER_TEXT and name == "__init__")
            ):
                continue
            if isinstance(value, property) or not callable(value):
                continue
            return_type = overrides.get(name, "Any")
            methods.append(_format_manual_method(name, value, return_type))
        return methods

    @staticmethod
    def _collect_plugin_methods(class_name: str, runtime: dict[str, Any]) -> list[str]:
        registry = runtime["registry"]
        texture_kind = runtime["TextureKind"]
        namespace_kinds = {
            _SDFNAMESPACE_TEXT: texture_kind.SDF,
            _COLORNAMESPACE_TEXT: texture_kind.COLOR,
        }
        if class_name in namespace_kinds:
            plugins = [
                registry.get(name)
                for name in registry.factory_names_for(namespace_kinds[class_name])
            ]
        else:
            plugins = [
                plugin
                for plugin in registry._plugins.values()
                if _plugin_matches_class(class_name, plugin, texture_kind)
            ]
        plugins.sort(key=lambda plugin: plugin.name)
        return [
            _format_plugin_method(plugin, _plugin_return_type(texture_kind, plugin))
            for plugin in plugins
        ]

    @staticmethod
    def _plugin_matches_class(class_name, plugin, texture_kind):
        match class_name:
            case "SDFTexture":
                return texture_kind.SDF in plugin.method_of and plugin.name not in {
                    "repeat",
                    "repeat_grid",
                }
            case "ColorTexture":
                return texture_kind.COLOR in plugin.method_of
            case "PostNamespace":
                return (
                    plugin.family == "postprocessing"
                    and texture_kind.COLOR in plugin.method_of
                )
            case _:
                return False

    @staticmethod
    def _merge_methods(class_name: str, cls: Any, runtime: dict[str, Any]) -> list[str]:
        manual = _collect_manual_methods(class_name, cls)
        existing_names = {
            line.split("def ", 1)[1].split("(", 1)[0]
            for line in manual
            if line.lstrip().startswith("def ")
        }

        merged = list(manual)
        for line in _collect_plugin_methods(class_name, runtime):
            name = line.split("def ", 1)[1].split("(", 1)[0]
            if name not in existing_names:
                merged.append(line)
                existing_names.add(name)
        return merged

    @staticmethod
    def _render_texture_stub(runtime: dict[str, Any]) -> str:
        lines = [
            _FROM___FUTURE___IMPORT_ANNOTATIONS_TEXT,
            "",
            "from collections.abc import Iterable, Mapping",
            "from pathlib import Path",
            "from typing import TYPE_CHECKING, Any, Iterator, Optional, Self, overload",
            "",
            "from numpy.typing import NDArray",
            "",
            "from .spatial import TextureChannel",
            "",
            "if TYPE_CHECKING:",
            "    from .color import ColorTexture",
            "",
            _render_class(
                _TEXTURENODE_TEXT,
                _collect_manual_methods(_TEXTURENODE_TEXT, runtime[_TEXTURENODE_TEXT]),
                properties=[
                    "    kind: str | None",
                    "    tex: Any",
                    "    context: Any",
                    "    op: str | None",
                    "    inputs: tuple[Any, ...]",
                    "    params: Any",
                    "    label: str | None",
                    "    should_cache: bool",
                    "    size: tuple[int, int]",
                    "    origin: tuple[int, int]",
                    "    canvas_size: tuple[int, int]",
                    "    @overload",
                    "    def __init__(self, tex: Any, context: Any, *, op: str | None = ..., inputs: Iterable[Any] = ..., params: Mapping[str, Any] | Iterable[Any] | None = ..., label: str | None = ..., should_cache: bool = ...) -> None: ...",
                    "    @overload",
                    "    def __init__(self, *, op: str, inputs: Iterable[Any] = ..., params: Mapping[str, Any] | Iterable[Any] | None = ..., label: str | None = ..., should_cache: bool = ...) -> None: ...",
                ],
            ).rstrip(),
            "",
            "class WarpableTextureNode(TextureNode):",
            "    def warp(self, field: Any, strength: Any = 1.0, steps: Any = 1) -> Self: ...",
            "",
            "_SPATIAL_PARAMS: set[str]",
            "def _scale_spatial_params(op: Any, params: Any, scale: Any) -> Any: ...",
            "",
            _render_class(
                _POSTNAMESPACE_TEXT,
                _merge_methods(
                    _POSTNAMESPACE_TEXT, runtime[_POSTNAMESPACE_TEXT], runtime
                ),
            ).rstrip(),
            "",
            _render_class(
                "RenderStats",
                [
                    "    shader_dispatches: int",
                    "    shader_dispatches_by_name: dict[str, int]",
                    "    cache_hits: int",
                    "    cache_misses: int",
                    "    cache_writes: int",
                    "    cache_skips: int",
                    "    texture_allocations: int",
                    "    texture_allocations_by_kind: dict[str, int]",
                    "    texture_reuses: int",
                    "    render_calls: int",
                    "    elapsed_seconds: float",
                    "    def as_dict(self) -> dict[str, Any]: ...",
                ],
            ).rstrip(),
            "",
            _render_class(
                "RenderCacheInfo",
                [
                    "    entries: int",
                    "    hits: int",
                    "    misses: int",
                    "    writes: int",
                    "    skips: int",
                ],
            ).rstrip(),
            "",
            _render_class(
                _RENDERER_TEXT,
                _collect_manual_methods(_RENDERER_TEXT, runtime[_RENDERER_TEXT]),
                properties=["    stats: RenderStats"],
            ).rstrip(),
            "",
            _render_class(
                "MultiOutputResult",
                [
                    "    op: str",
                    "    inputs: tuple[Any, ...]",
                    "    params: tuple[Any, ...]",
                    "    label: Optional[str]",
                    "    should_cache: bool",
                    "    kind: str",
                    "    def render(self, ctx: Any, params: Optional[dict[str, Any]] = ..., cache: Any = ...) -> Any: ...",
                    "    def __iter__(self) -> Iterator[Any]: ...",
                ],
            ).rstrip(),
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_color_stub(runtime: dict[str, Any]) -> str:
        methods = [
            "    kind: str",
            "    mode: str",
            "    @overload",
            "    def __init__(self, tex: Any, context: Any, *, mode: str = ..., op: str | None = ..., inputs: Any = ..., params: Any = ..., label: str | None = ..., should_cache: bool = ...) -> None: ...",
            "    @overload",
            "    def __init__(self, *, op: str, mode: str = ..., inputs: Any = ..., params: Any = ..., label: str | None = ..., should_cache: bool = ...) -> None: ...",
            "    @property",
            "    def post(self) -> PostNamespace: ...",
        ]
        methods.extend(
            _merge_methods(_COLORTEXTURE_TEXT, runtime[_COLORTEXTURE_TEXT], runtime)
        )
        lines = [
            _FROM___FUTURE___IMPORT_ANNOTATIONS_TEXT,
            "",
            "from typing import Any, overload",
            "",
            "from .texture import PostNamespace, WarpableTextureNode",
            "",
            "class ColorSpaceMode:",
            "    LAB: str",
            "    RGB: str",
            "    LINEAR: str",
            "",
            _render_class("ColorTexture(WarpableTextureNode)", methods).rstrip(),
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_field_stub(runtime: dict[str, Any]) -> str:
        methods = ["    kind: str"]
        methods.extend(
            _merge_methods(_FIELDTEXTURE_TEXT, runtime[_FIELDTEXTURE_TEXT], runtime)
        )
        lines = [
            _FROM___FUTURE___IMPORT_ANNOTATIONS_TEXT,
            "",
            "from typing import Any",
            "",
            "from .color import ColorTexture",
            "from .texture import TextureNode",
            "",
            _render_class("FieldTexture(TextureNode)", methods).rstrip(),
            "",
            _render_class(
                _FIELDNAMESPACE_TEXT,
                _collect_manual_methods(
                    _FIELDNAMESPACE_TEXT, runtime[_FIELDNAMESPACE_TEXT]
                ),
            ).rstrip(),
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_sdf_stub(runtime: dict[str, Any]) -> str:
        methods = ["    kind: str"]
        methods.extend(
            _merge_methods(_SDFTEXTURE_TEXT, runtime[_SDFTEXTURE_TEXT], runtime)
        )
        lines = [
            _FROM___FUTURE___IMPORT_ANNOTATIONS_TEXT,
            "",
            "from typing import Any",
            "",
            "from .color import ColorTexture",
            "from .texture import MultiOutputResult, WarpableTextureNode",
            "",
            _render_class("SDFTexture(WarpableTextureNode)", methods).rstrip(),
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_api_stub(runtime: dict[str, Any]) -> str:
        sdf_methods = _merge_methods(
            _SDFNAMESPACE_TEXT, runtime[_SDFNAMESPACE_TEXT], runtime
        )
        color_methods = _merge_methods(
            _COLORNAMESPACE_TEXT, runtime[_COLORNAMESPACE_TEXT], runtime
        )
        lines = [
            _FROM___FUTURE___IMPORT_ANNOTATIONS_TEXT,
            "",
            "from collections.abc import Iterable",
            "from typing import Any",
            "",
            "from .color import ColorTexture",
            "from .palette import Palette",
            "from .sdf import SDFTexture",
            "from .texture import MultiOutputResult",
            "",
            _render_class(_SDFNAMESPACE_TEXT, sdf_methods).rstrip(),
            "",
            _render_class(_COLORNAMESPACE_TEXT, color_methods).rstrip(),
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_core_init_stub() -> str:
        return "\n".join(
            [
                "from .color import ColorSpaceMode as ColorSpaceMode, ColorTexture as ColorTexture",
                "from .expressions import Expr as Expr, cos as cos, input_value as input_value, param as param, percent as percent, percent_of_min as percent_of_min, percent_x as percent_x, percent_y as percent_y, sin as sin",
                "from .field import FieldNamespace as FieldNamespace, FieldTexture as FieldTexture",
                "from .palette import Palette as Palette, get_palette as get_palette, palettes as palettes",
                "from .sdf import SDFTexture as SDFTexture",
                "",
            ]
        )

    @staticmethod
    def _render_root_init_stub() -> str:
        return "\n".join(
            [
                "from typing import Any",
                "",
                "from . import core as core, learning as learning",
                "from .gpu.context import Context as Context, DispatchConfig as DispatchConfig",
                "from .core import ColorTexture as ColorTexture, FieldTexture as FieldTexture, SDFTexture as SDFTexture",
                "from .core.api import ColorNamespace, SDFNamespace",
                "from .core.field import FieldNamespace",
                "from .core.palette import Palette as Palette, get_palette as get_palette, palettes as palettes",
                "from .core.expressions import Expr as Expr, cos as cos, ease_in_cubic as ease_in_cubic, ease_in_out_sine as ease_in_out_sine, ease_out_cubic as ease_out_cubic, input_value as input_value, param as param, percent as percent, percent_of_min as percent_of_min, percent_x as percent_x, percent_y as percent_y, sin as sin",
                "from .plot import Plot as Plot",
                "from .text import badge as badge, glyph as glyph, icon as icon",
                "",
                "Canvas = Context",
                "sdf: SDFNamespace",
                "color: ColorNamespace",
                "field: FieldNamespace",
                "logger: Any",
                "",
            ]
        )


_render_class = _StubRenderer._render_class
_plugin_return_type = _StubRenderer._plugin_return_type
_method_return_overrides = _StubRenderer._method_return_overrides
_collect_manual_methods = _StubRenderer._collect_manual_methods
_collect_plugin_methods = _StubRenderer._collect_plugin_methods
_plugin_matches_class = _StubRenderer._plugin_matches_class
_merge_methods = _StubRenderer._merge_methods
_render_texture_stub = _StubRenderer._render_texture_stub
_render_color_stub = _StubRenderer._render_color_stub
_render_field_stub = _StubRenderer._render_field_stub
_render_sdf_stub = _StubRenderer._render_sdf_stub
_render_api_stub = _StubRenderer._render_api_stub
_render_core_init_stub = _StubRenderer._render_core_init_stub
_render_root_init_stub = _StubRenderer._render_root_init_stub


def generate_api_stubs(*, check: bool = False) -> bool:
    """Generate tracked stubs, or report stale files without changing them."""
    runtime = _load_runtime_objects()

    outputs = {
        CORE / "texture.pyi": _render_texture_stub(runtime),
        CORE / "color.pyi": _render_color_stub(runtime),
        CORE / "sdf.pyi": _render_sdf_stub(runtime),
        CORE / "field.pyi": _render_field_stub(runtime),
        CORE / "api.pyi": _render_api_stub(runtime),
        CORE / "__init__.pyi": _render_core_init_stub(),
        PACKAGE / "__init__.pyi": _render_root_init_stub(),
    }

    stale = []
    for path, content in outputs.items():
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            stale.append(path)
            if not check:
                _write_if_changed(path, content)
    if check and stale:
        for path in stale:
            print(f"Stale generated stub: {path}", file=sys.stderr)
        print("Run uv run python build_tools/generate_api_stubs.py", file=sys.stderr)
    return not stale if check else True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or check public API stubs")
    parser.add_argument(
        "--check", action="store_true", help="report stale stubs without writing"
    )
    args = parser.parse_args()
    if not generate_api_stubs(check=args.check) and args.check:
        raise SystemExit(1)
