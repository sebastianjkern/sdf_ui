"""Rounded containers defined by a full canvas box."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

from ..core.color import ColorTexture
from ..core.expressions import _lazy_scalar, _scalar_minimum
from ..core.sdf import SDFTexture

_TRANSPARENT = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Panel:
    """A rounded rectangle with a padded content box.

    Use lower-left canvas coordinates and full dimensions ``(x, y, width, height)``.
    Padding is ``(horizontal, vertical)`` on each side. Dimensions, padding,
    and corner radius can be deferred expressions.
    """

    box: tuple[Any, Any, Any, Any]
    padding: tuple[Any, Any] = (0.0, 0.0)
    radius: Any = 0.0

    def __post_init__(self):
        x, y, width, height = self.box
        horizontal, vertical = self.padding
        for name, value in (
            ("x", x),
            ("y", y),
            ("width", width),
            ("height", height),
            ("horizontal padding", horizontal),
            ("vertical padding", vertical),
            ("radius", self.radius),
        ):
            if isinstance(value, Real):
                if not math.isfinite(value):
                    raise ValueError(f"panel {name} must be finite")
                if name not in {"x", "y"} and value < 0:
                    raise ValueError(f"panel {name} must be nonnegative")
        object.__setattr__(self, "box", (x, y, width, height))
        object.__setattr__(self, "padding", (horizontal, vertical))

    @property
    def content_box(self) -> tuple[Any, Any, Any, Any]:
        """Return the inset box in canvas coordinates for child layout."""
        x, y, width, height = map(_lazy_scalar, self.box)
        horizontal, vertical = map(_lazy_scalar, self.padding)
        horizontal = _scalar_minimum(horizontal, width / 2)
        vertical = _scalar_minimum(vertical, height / 2)
        return (
            x + horizontal,
            y + vertical,
            width - 2 * horizontal,
            height - 2 * vertical,
        )

    def shape(self) -> SDFTexture:
        """Return the geometry for masks, styling, or field losses."""
        from .. import sdf

        x, y, width, height = map(_lazy_scalar, self.box)
        half_width, half_height = width / 2, height / 2
        radius = _scalar_minimum(self.radius, _scalar_minimum(half_width, half_height))
        return sdf.rounded_rect(
            (x + half_width, y + half_height), (half_width, half_height), radius
        )

    def layer(
        self, background, content: ColorTexture | None = None, *, clip: bool = False
    ) -> ColorTexture:
        """Paint the panel and optionally overlay content in canvas coordinates.

        Use ``content_box`` to position a child and ``clip=True`` to clip it
        to the outer rounded shape.
        """
        shape = self.shape()
        surface = shape.fill(background, _TRANSPARENT)
        if content is None:
            return surface
        if clip:
            content = shape.fill_from_texture(
                content, background=_TRANSPARENT, inflate=-3 / 4
            )
        return surface.alpha_overlay(content)
