from __future__ import annotations

from pathlib import Path
from typing import Any

from .core.sdf import SDFTexture

def glyph(
    char: str,
    scale: Any,
    ox: Any,
    oy: Any,
    path: str | Path | None = ...,
    samples: int = ...,
) -> SDFTexture: ...

def text(
    value: Any,
    size: Any = ...,
    ox: Any = ...,
    oy: Any = ...,
    path: str | Path | None = ...,
    samples: int = ...,
    cache_size: int = ...,
    line_height: float = ...,
    oversample: float = ...,
    min_render_size: float = ...,
    box: tuple[float, float, float, float] | None = ...,
    align: str = ...,
    valign: str = ...,
    fallback_fonts: str | Path | tuple[str | Path, ...] | list[str | Path] = ...,
    missing_glyph: str = ...,
    clip: bool = ...,
    overflow: str = ...,
    trim_trailing_spaces: bool = ...,
    direction: str | None = ...,
    script: str | None = ...,
    language: str | None = ...,
    features: dict[str, int | bool] | None = ...,
    variations: dict[str, float] | None = ...,
    white_space: str = ...,
    first_line_indent: float = ...,
    max_lines: int | None = ...,
    decoration: str | tuple[str, ...] = ...,
    decoration_thickness: float = ...,
    word_spacing: float = ...,
    letter_spacing: float = ...,
    vertical_metrics: str = ...,
) -> SDFTexture: ...

def span(value: str, **style: Any) -> dict[str, Any]: ...

def measure_text(
    value: str,
    size: float = ...,
    path: str | Path | None = ...,
    *,
    width: float | None = ...,
    height: float | None = ...,
    line_height: float = ...,
    fallback_fonts: str | Path | tuple[str | Path, ...] | list[str | Path] = ...,
    missing_glyph: str = ...,
    direction: str | None = ...,
    script: str | None = ...,
    language: str | None = ...,
    features: dict[str, int | bool] | None = ...,
    variations: dict[str, float] | None = ...,
    trim_trailing_spaces: bool = ...,
    white_space: str = ...,
    word_spacing: float = ...,
    letter_spacing: float = ...,
    first_line_indent: float = ...,
    max_lines: int | None = ...,
    vertical_metrics: str = ...,
) -> dict[str, Any]: ...

def measure_glyph(
    char: str,
    size: float = ...,
    path: str | Path | None = ...,
    *,
    fallback_fonts: str | Path | tuple[str | Path, ...] | list[str | Path] = ...,
    variations: dict[str, float] | None = ...,
) -> dict[str, Any]: ...

def system_font(family: str = ..., style: str = ...) -> str: ...
