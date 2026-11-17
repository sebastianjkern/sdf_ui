"""Reusable color palettes and interpolation helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import TypeAlias

from ..util import hex_col

Color: TypeAlias = tuple[float, float, float, float]


def _normalize_color(value) -> Color:
    if isinstance(value, str):
        channels = hex_col(value)
    else:
        try:
            channels = tuple(float(channel) for channel in value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "palette colors must be hex strings or RGB(A) values"
            ) from error
        if len(channels) == 3:
            channels += (1.0,)
    if len(channels) != 4 or any(
        not math.isfinite(channel) or not 0.0 <= channel <= 1.0 for channel in channels
    ):
        raise ValueError("palette colors must have 3 or 4 channels in [0, 1]")
    return tuple(channels)


@dataclass(frozen=True)
class Palette:
    """A named or custom palette with normalized color sampling.

    ``sample`` linearly interpolates adjacent entries for a position in [0, 1].
    Colors can be ``#RRGGBB``/``#RRGGBBAA`` strings or normalized RGB(A) tuples.
    """

    colors: tuple[Color, ...]
    name: str | None = None

    def __post_init__(self):
        try:
            values = tuple(_normalize_color(value) for value in self.colors)
        except TypeError as error:
            raise ValueError("a palette needs at least two colors") from error
        if len(values) < 2:
            raise ValueError("a palette needs at least two colors")
        object.__setattr__(self, "colors", values)

    def sample(self, position: Real) -> Color:
        """Return an interpolated RGBA color at normalized ``position``."""
        if isinstance(position, bool) or not isinstance(position, Real):
            raise ValueError("palette position must be a finite number in [0, 1]")
        position = float(position)
        if not math.isfinite(position) or not 0.0 <= position <= 1.0:
            raise ValueError("palette position must be a finite number in [0, 1]")
        scaled = position * (len(self.colors) - 1)
        index = min(int(scaled), len(self.colors) - 2)
        amount = scaled - index
        start, end = self.colors[index], self.colors[index + 1]
        return tuple(first + (last - first) * amount for first, last in zip(start, end))

    def steps(self, count: int) -> tuple[Color, ...]:
        """Sample ``count`` evenly spaced colors, including both endpoints."""
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError("count must be a positive integer")
        count = int(count)
        if count == 1:
            return (self.sample(0.5),)
        return tuple(self.sample(index / (count - 1)) for index in range(count))

    def reversed(self) -> Palette:
        """Return the same palette in reverse color order."""
        return Palette(tuple(reversed(self.colors)), name=self.name)


_BUILTIN_PALETTES = {
    "starry_night": Palette(
        ("#071638", "#142d68", "#3266a3", "#75b5d1", "#f0c768"),
        name="starry_night",
    ),
    "ember": Palette(
        ("#24112d", "#702b47", "#c05245", "#ed934f", "#f7d47c"),
        name="ember",
    ),
    "sea_glass": Palette(
        ("#082c45", "#116878", "#41a69a", "#a6d6b8", "#f1e5b2"),
        name="sea_glass",
    ),
    "moss": Palette(
        ("#172a2b", "#315344", "#698253", "#b2aa69", "#e9d59a"),
        name="moss",
    ),
}
palettes = MappingProxyType(_BUILTIN_PALETTES)


def get_palette(value: str | Palette | Iterable) -> Palette:
    """Resolve a built-in palette name or normalize a custom color sequence."""
    if isinstance(value, Palette):
        return value
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        try:
            return palettes[key]
        except KeyError as error:
            choices = ", ".join(palettes)
            raise ValueError(
                f"Unknown palette {value!r}; choose from {choices}"
            ) from error
    try:
        return Palette(tuple(value))
    except TypeError as error:
        raise ValueError(
            "palette must be a built-in name or a color sequence"
        ) from error
