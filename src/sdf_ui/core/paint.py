"""Paint shared SDF scenes as pixels or pen paths."""

from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

from .._hatch_geometry import HatchFrame, StrokeDrift, relief_shadow, stroke_lines
from ..gpu.nodes.common import build
from ..gpu.nodes.common import color as normalize_color
from ..vector import VectorDrawing, VectorPath, contour_lines, hatch_lines
from .color import ColorTexture
from .sdf import SDFTexture

if TYPE_CHECKING:
    from ..export import PlotterProfile
    from ..gpu.context import Context
    from ..learning.autodiff import Tensor
    from .expressions import Expr
    from .texture import Renderer

Scalar = Union[float, str, "Expr", "Tensor"]
Ink = Union[
    str,
    tuple[Scalar, Scalar, Scalar],
    tuple[Scalar, Scalar, Scalar, Scalar],
    "Expr",
    "Tensor",
]
PaintOperation = Literal["fill", "outline", "isolines"]


@dataclass(frozen=True)
class HatchPass:
    """A family of parallel strokes. Angle is in radians, lengths in canvas units."""

    angle: Scalar
    spacing: Scalar
    width: Scalar = 1.0
    phase: Scalar = 0.0
    min_shadow: Scalar = 0.0

    def __post_init__(self):
        _validate_literals(
            angle=self.angle,
            spacing=self.spacing,
            width=self.width,
            phase=self.phase,
            min_shadow=self.min_shadow,
        )


@dataclass(frozen=True)
class HandStroke:
    """Variation between strokes, in canvas units, with one gentle arc per stroke."""

    bow: Scalar = 3.0
    jitter: Scalar = 1.0
    seed: int = 0

    def __post_init__(self):
        _validate_literals(bow=self.bow, jitter=self.jitter)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Hand stroke seed must be a nonnegative integer")


@dataclass(frozen=True)
class HatchLight:
    """Directional light on a rounded SDF relief. Axes are right, up and forward."""

    direction: tuple[Scalar, Scalar, Scalar] = (-1.0, 1.0, 1.0)
    depth: Scalar = 60.0

    def __post_init__(self):
        if len(self.direction) != 3:
            raise ValueError("Hatch light direction requires three coordinates")
        _validate_literals(depth=self.depth)


@dataclass(frozen=True)
class HatchStyle:
    """Ordered hatch passes sharing an origin in logical canvas coordinates."""

    passes: tuple[HatchPass, ...]
    origin: tuple[Scalar, Scalar] = (0.0, 0.0)
    hand: HandStroke | None = None
    light: HatchLight | None = None

    def __post_init__(self):
        if len(self.origin) != 2:
            raise ValueError("A hatch style origin requires two coordinates")
        if not self.passes or not all(
            isinstance(item, HatchPass) for item in self.passes
        ):
            raise ValueError("A hatch style requires at least one HatchPass")
        _validate_literals(origin_x=self.origin[0], origin_y=self.origin[1])
        if self.light is None and any(
            not isinstance(item.min_shadow, (int, float)) or item.min_shadow != 0
            for item in self.passes
        ):
            raise ValueError("A hatch shadow threshold requires HatchLight")

    @property
    def _uses_paths(self) -> bool:
        return self.hand is not None or self.light is not None


def _validate_literals(**values):
    for name, value in values.items():
        if isinstance(value, (int, float)):
            _checked_scalar(name, value)


def _checked_scalar(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must resolve to a finite number")
    if name in {"spacing", "width", "depth"} and value <= 0:
        raise ValueError(f"{name} must be positive")
    if name in {"bow", "jitter"} and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if name == "min_shadow" and not 0 <= value <= 1:
        raise ValueError("min_shadow must be in [0, 1]")
    return float(value)


@dataclass(frozen=True)
class PaintLayer:
    shape: SDFTexture
    operation: PaintOperation
    color: Ink
    inflate: Scalar = 0.0
    hatch: HatchStyle | None = None
    pen: str | None = None
    width: Scalar = 1.5
    spacing: Scalar = 12.0
    phase: Scalar = 0.0

    def __post_init__(self):
        if not isinstance(self.shape, SDFTexture):
            raise TypeError("A paint layer requires an SDFTexture")
        _validate_literals(
            inflate=self.inflate,
            width=self.width,
            spacing=self.spacing,
            phase=self.phase,
        )

    def _raster(self) -> ColorTexture:
        if self.operation == "outline":
            return self.shape.outline(
                self.color, inflate=self.inflate, width=self.width
            )
        if self.operation == "isolines":
            return self.shape.isolines(
                self.color,
                inflate=self.inflate,
                spacing=self.spacing,
                line_width=self.width,
                phase=self.phase,
            )
        if self.hatch is None:
            return self.shape.ink(self.color, inflate=self.inflate)
        origin_x, origin_y = self.hatch.origin
        layers = [
            build(
                "hatch_fill",
                self.shape,
                fg_color=self.color,
                inflate=self.inflate,
                angle=item.angle,
                spacing=item.spacing,
                phase=item.phase,
                width=item.width,
                origin_x=origin_x,
                origin_y=origin_y,
            )
            for item in self.hatch.passes
        ]
        return layers[0].over(*layers[1:])

    def _paths(self, pixels, renderer: Renderer, size: tuple[int, int], scale: float):
        def resolve(name, value):
            return _checked_scalar(name, renderer.render(value))

        inflate = resolve("inflate", self.inflate)
        rgba = normalize_color(renderer.render(self.color))
        if len(rgba) == 3:
            rgba = (*rgba, 1.0)
        channels = tuple(round(float(np.clip(channel, 0, 1)) * 255) for channel in rgba)
        if channels[3] == 0:
            return
        ink = "#" + "".join(f"{channel:02x}" for channel in channels)
        ink = ink[:7] if channels[3] == 255 else ink
        height, width_px = pixels.shape
        scale_x, scale_y = size[0] / width_px, size[1] / height
        for lines, width in self._geometry(pixels, inflate * scale, resolve, scale):
            for line in lines:
                points = tuple(
                    (
                        min(size[0], max(0.0, (x + 0.5) * scale_x)),
                        min(size[1], max(0.0, (height - 0.5 - y) * scale_y)),
                    )
                    for x, y in line
                )
                yield VectorPath(points, ink, width, self.pen)

    def _geometry(self, pixels, inflate: float, resolve, scale: float):
        if self.operation == "fill":
            hatch = self.hatch or _SOLID_PLOT_STYLE
            yield from _fill_geometry(pixels, inflate, hatch, resolve, scale)
        else:
            width = resolve("width", self.width)
            for level in self._levels(pixels, inflate, resolve, scale):
                yield contour_lines(pixels, level=level), width

    def _levels(self, pixels, inflate: float, resolve, scale: float):
        if self.operation == "outline":
            return (inflate,)
        return _isoline_levels(
            pixels,
            inflate,
            resolve("spacing", self.spacing) * scale,
            resolve("phase", self.phase) * scale,
        )


def _fill_geometry(pixels, inflate, hatch, resolve, scale):
    x, y = hatch.origin
    origin = (resolve("origin", x) * scale, resolve("origin", y) * scale)
    drift = _resolve_drift(hatch.hand, resolve, scale)
    depth, shadow = _resolve_shadow(pixels, inflate, hatch.light, resolve, scale)
    for item in hatch.passes:
        angle = resolve("angle", item.angle)
        spacing = resolve("spacing", item.spacing) * scale
        phase = (resolve("phase", item.phase) * scale) % spacing
        if hatch._uses_paths:
            threshold = resolve("min_shadow", item.min_shadow)
            mask = pixels - inflate
            if shadow is not None and threshold > 0:
                mask = np.maximum(mask, depth * (threshold - shadow))
            lines = stroke_lines(mask, HatchFrame(angle, spacing, phase, origin), drift)
        else:
            lines = hatch_lines(
                pixels,
                level=inflate,
                angle=angle,
                spacing=spacing,
                phase=phase,
                origin=origin,
            )
        yield lines, resolve("width", item.width)


def _resolve_drift(hand, resolve, scale):
    if hand is None:
        return None
    return StrokeDrift(
        resolve("bow", hand.bow) * scale,
        resolve("jitter", hand.jitter) * scale,
        hand.seed,
    )


def _resolve_shadow(pixels, inflate, light, resolve, scale):
    if light is None:
        return 0.0, None
    direction = tuple(resolve("direction", value) for value in light.direction)
    if math.hypot(*direction) == 0:
        raise ValueError("Hatch light direction must be nonzero")
    depth = resolve("depth", light.depth) * scale
    return depth, relief_shadow(pixels, inflate, direction, depth)


_SOLID_PLOT_STYLE = HatchStyle((HatchPass(0.0, 8.0),))


def _isoline_levels(pixels, inflate: float, spacing: float, phase: float):
    finite = pixels[np.isfinite(pixels)]
    if not finite.size:
        return ()
    low, high = float(finite.min()), float(finite.max())
    inner = max(0.0, low - inflate, inflate - high)
    outer = max(abs(low - inflate), abs(high - inflate))
    phase %= spacing
    first = max(0, math.ceil((inner + phase) / spacing))
    last = math.floor((outer + phase) / spacing)
    if last - first > 10_000:
        raise ValueError("Too many isolines for the requested spacing")
    levels: set[float] = set()
    for index in range(first, last + 1):
        distance = index * spacing - phase
        levels.update(
            level
            for level in (inflate - distance, inflate + distance)
            if low <= level <= high
        )
    return tuple(sorted(levels))


@dataclass(frozen=True)
class PaintScene:
    """An ordered scene that can be rendered as raster color or pen paths."""

    layers: tuple[PaintLayer, ...]
    background: Ink = (0.0, 0.0, 0.0, 0.0)
    _color: ColorTexture | None = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        raster = None
        if not any(layer.hatch and layer.hatch._uses_paths for layer in self.layers):
            background = build("clear", self.background)
            raster = background.over(*(layer._raster() for layer in self.layers))
        object.__setattr__(self, "_color", raster)

    def to_color(self) -> ColorTexture:
        """Return the existing color graph for rendering or training."""
        if self._color is None:
            raise ValueError(
                "Hand strokes and hatch lighting use sampled paths. Render with save_png()."
            )
        return self._color

    def to_vector(
        self,
        *,
        canvas: Context,
        params: dict | None = None,
        progress: float = 0.0,
    ) -> VectorDrawing:
        """Trace the current fields at the canvas sampling resolution."""
        size = canvas.logical_size
        with (
            canvas.session(params=params, progress=progress) as renderer,
            ExitStack() as results,
        ):
            fields = {}
            for layer in self.layers:
                key = id(layer.shape)
                if key not in fields:
                    result = renderer.render(layer.shape)
                    if result is not layer.shape:
                        results.enter_context(result)
                    fields[key] = result.read()
            paths = tuple(
                path
                for layer in self.layers
                for path in layer._paths(
                    fields[id(layer.shape)], renderer, size, canvas.device_scale
                )
            )
        return VectorDrawing(size, paths)

    def save_png(
        self,
        path: str | Path,
        *,
        canvas: Context,
        params: dict | None = None,
        progress: float = 0.0,
    ) -> Path:
        if self._color is not None:
            return canvas.save(self._color, path, params, progress=progress)
        from ._paint_raster import save_path_scene

        return save_path_scene(self, path, canvas, params, progress)

    def save_svg(
        self,
        path: str | Path,
        *,
        canvas: Context,
        params: dict | None = None,
        progress: float = 0.0,
    ) -> Path:
        from ..export import write_svg

        return write_svg(
            self.to_vector(canvas=canvas, params=params, progress=progress), path
        )

    def save_gcode(
        self,
        path: str | Path,
        *,
        canvas: Context,
        plotter: PlotterProfile,
        params: dict | None = None,
        progress: float = 0.0,
    ) -> Path:
        from ..export import write_gcode

        drawing = self.to_vector(canvas=canvas, params=params, progress=progress)
        return write_gcode(drawing, path, plotter=plotter)


def fill(
    shape: SDFTexture,
    color: Ink,
    *,
    inflate: Scalar = 0.0,
    hatch: HatchStyle | None = None,
    pen: str | None = None,
) -> PaintLayer:
    """Paint an SDF interior. Without a hatch recipe, plots use parallel strokes."""
    return PaintLayer(shape, "fill", color, inflate, hatch, pen)


def outline(
    shape: SDFTexture,
    color: Ink,
    *,
    inflate: Scalar = 0.0,
    width: Scalar = 1.5,
    pen: str | None = None,
) -> PaintLayer:
    """Paint one stroke centered on the SDF boundary."""
    return PaintLayer(shape, "outline", color, inflate, pen=pen, width=width)


def isolines(
    shape: SDFTexture,
    color: Ink,
    *,
    inflate: Scalar = 0.0,
    spacing: Scalar = 12.0,
    width: Scalar = 1.5,
    phase: Scalar = 0.0,
    pen: str | None = None,
) -> PaintLayer:
    """Paint repeated distance contours inside and outside the SDF boundary."""
    return PaintLayer(
        shape,
        "isolines",
        color,
        inflate,
        pen=pen,
        width=width,
        spacing=spacing,
        phase=phase,
    )


def scene(*layers: PaintLayer, background: Ink = (0.0, 0.0, 0.0, 0.0)) -> PaintScene:
    """Compose layers in drawing order with an optional raster paper color."""
    return PaintScene(layers, background)
