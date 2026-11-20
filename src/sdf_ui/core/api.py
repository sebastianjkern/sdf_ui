"""Dynamic namespaces that expose plugin-backed factory functions."""

from __future__ import annotations

__docformat__ = "google"

import math
from collections.abc import Iterable
from numbers import Real
from typing import TYPE_CHECKING

from ..gpu.nodes.base import TextureKind
from ..gpu.nodes.common import build
from ._suggestions import unknown_name_message
from .color import _transparent_ink
from .expressions import (
    _lazy_scalar,
    _scalar_maximum,
    _scalar_minimum,
    _select_positive,
    cos,
    sin,
)

if TYPE_CHECKING:
    from .sdf import SDFTexture


def _half_dimensions(size):
    width, height = (_lazy_scalar(value) for value in size)
    return width / 2, height / 2


def _rect_with_size(factory, center, size, corner_radius, angle):
    center = tuple(_lazy_scalar(value) for value in center)
    return build(
        factory,
        center,
        _half_dimensions(size),
        corner_radius=corner_radius,
        angle=angle,
    )


def _arrow_head(ax, ay, bx, by, radius, head_length, half_head_width):
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    direction = length_sq - 1 / 1_000_000
    if isinstance(direction, Real) and direction <= 0:
        return build("capsule", (ax, ay), (bx, by), radius)
    length = _scalar_maximum(length_sq, 1 / 1_000_000) ** 0.5
    ux = _select_positive(direction, dx / length, 1.0)
    uy = _select_positive(direction, dy / length, 0.0)
    fitted_head = _scalar_minimum(head_length, length)
    tip_length = _select_positive(direction, fitted_head, radius / 2)
    tip_half_width = _select_positive(
        direction, half_head_width * (fitted_head / head_length), radius / 2
    )
    base_x, base_y = bx - ux * tip_length, by - uy * tip_length
    shaft_end = (
        _select_positive(direction, base_x, bx),
        _select_positive(direction, base_y, by),
    )
    shaft = build("capsule", (ax, ay), shaft_end, radius)
    head = build(
        "triangle",
        (bx, by),
        (base_x - uy * tip_half_width, base_y + ux * tip_half_width),
        (base_x + uy * tip_half_width, base_y - ux * tip_half_width),
    )
    return shaft.union(head)


def _validate_ring_dimensions(count, line_width, gap):
    import math
    from numbers import Integral

    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError("concentric ring count must be a positive integer")
    for name, value, allow_zero in (
        ("line_width", line_width, False),
        ("gap", gap, True),
    ):
        _validate_ring_measure(name, value, allow_zero, math)


def _validate_ring_measure(name, value, allow_zero, math):
    if not isinstance(value, Real):
        return
    invalid = (
        isinstance(value, bool) or not math.isfinite(value) or value < 0
        or (not allow_zero and value == 0)
    )
    if invalid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"concentric ring {name} must be finite and {qualifier}")


class _OperationNamespace:
    kind: str | None = None

    def __getattr__(self, name):
        from ..gpu.nodes.registry import registry

        try:
            plugin = registry.get(name)
        except KeyError as exc:
            raise AttributeError(self._unknown_factory_message(name)) from exc
        if not plugin.public or plugin.result != self.kind or plugin.input_kinds:
            raise AttributeError(self._unknown_factory_message(name))

        def factory(*args, **kwargs):
            return build(name, *args, **kwargs)

        factory.__name__ = name
        factory.__qualname__ = f"{type(self).__name__}.{name}"
        factory.__doc__ = plugin.docstring()
        factory.__signature__ = plugin.signature()
        return factory

    def _unknown_factory_message(self, name):
        from ..gpu.nodes.registry import registry

        names = registry.factory_names_for(self.kind)
        return unknown_name_message(f"No {self.kind} factory named", name, names)

    def __dir__(self):
        from ..gpu.nodes.registry import registry

        names = set(super().__dir__())
        names.update(registry.factory_names_for(self.kind))
        return sorted(names)


class SDFNamespace(_OperationNamespace):
    kind = TextureKind.SDF

    def union_all(self, shapes: Iterable[SDFTexture]):
        """Combine an iterable of signed distance fields with hard union.

        The iterable must contain at least one SDF. Inputs stay as scene nodes,
        so lazy parameters and shared subgraphs remain available to rendering
        and learning.
        """
        from .sdf import SDFTexture

        iterator = iter(shapes)
        try:
            result = next(iterator)
        except StopIteration as exc:
            raise ValueError("union_all requires at least one shape") from exc
        if not isinstance(result, SDFTexture):
            raise TypeError("union_all expects SDFTexture instances")
        for shape in iterator:
            if not isinstance(shape, SDFTexture):
                raise TypeError("union_all expects SDFTexture instances")
            result = result.union(shape)
        return result

    def rect_from_size(self, center, size, corner_radius=0.0, angle=0.0):
        """Create a rectangle using full width and height measurements.

        ``size`` is the full width and height. The existing ``rect`` factory
        continues to take half extents. Center and size values may be lazy
        expressions or trainable parameters.
        """
        return _rect_with_size("rect", center, size, corner_radius, angle)

    def rounded_rect_from_size(self, center, size, corner_radius, angle=0.0):
        """Create a rounded rectangle using full width and height measurements.

        ``size`` is the full width and height. The existing ``rounded_rect``
        factory continues to take half extents. Center, size, and corner-radius
        values may be lazy expressions or trainable parameters.
        """
        return _rect_with_size("rounded_rect", center, size, corner_radius, angle)

    def star(
        self, center, outer_radius, inner_radius=None, rays: int = 5, rotation=-math.pi / 2
    ):
        """Create a regular star from differentiable triangular wedges.

        ``rays`` is a fixed integer of at least two. Center, radii, and rotation
        may be lazy expressions or trainable parameters. Radii must remain
        positive, with ``inner_radius`` smaller than ``outer_radius``. When
        omitted, the inner radius is 45 percent of the outer radius.
        """
        self._validate_star_parameters(outer_radius, inner_radius, rays, rotation)
        center_x, center_y = map(_lazy_scalar, center)
        outer_radius = _lazy_scalar(outer_radius)
        inner_radius = (
            outer_radius * (45 / 100)
            if inner_radius is None
            else _lazy_scalar(inner_radius)
        )
        rotation = _lazy_scalar(rotation)
        vertices = self._star_vertices(center_x, center_y, outer_radius, inner_radius, rays, rotation)
        wedges = (
            build(
                "triangle",
                (center_x, center_y),
                vertices[index],
                vertices[(index + 1) % len(vertices)],
            )
            for index in range(len(vertices))
        )
        return self.union_all(wedges)

    @staticmethod
    def _validate_star_parameters(outer_radius, inner_radius, rays, rotation):
        from numbers import Integral

        if isinstance(rays, bool) or not isinstance(rays, Integral) or rays < 2:
            raise ValueError("star rays must be an integer of at least 2")
        SDFNamespace._validate_star_radius(outer_radius, "outer_radius")
        SDFNamespace._validate_star_radius(inner_radius, "inner_radius")
        SDFNamespace._validate_star_rotation(rotation)
        if (
            isinstance(outer_radius, Real)
            and isinstance(inner_radius, Real)
            and inner_radius >= outer_radius
        ):
            raise ValueError("star inner_radius must be smaller than outer_radius")

    @staticmethod
    def _validate_star_radius(value, name):
        if value is None or not isinstance(value, Real):
            return
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"star {name} must be finite and positive")

    @staticmethod
    def _validate_star_rotation(rotation):
        if isinstance(rotation, Real) and (
            isinstance(rotation, bool) or not math.isfinite(rotation)
        ):
            raise ValueError("star rotation must be finite")

    @staticmethod
    def _star_vertices(center_x, center_y, outer_radius, inner_radius, rays, rotation):
        vertices = []
        for index in range(int(rays) * 2):
            angle = rotation + index * math.pi / int(rays)
            radius = outer_radius if index % 2 == 0 else inner_radius
            vertices.append(
                (
                    center_x + radius * cos(angle),
                    center_y + radius * sin(angle),
                )
            )
        return tuple(vertices)

    def fir_tree(self, center_x, base_y, height, width, tiers: int = 3):
        """Build a stepped fir silhouette from triangle boughs and a trunk.

        Position and dimensions accept lazy expressions or trainable parameters.
        ``tiers`` is a fixed integer topology control of at least one. A single
        tier is a plain triangle; larger values add stepped boughs and a trunk.
        """
        self._validate_fir_tree(tiers, height, width)
        center_x, base_y, height, width = map(
            _lazy_scalar, (center_x, base_y, height, width)
        )
        if tiers == 1:
            return self._single_fir_tree(center_x, base_y, height, width)
        return self.union_all(
            self._tiered_fir_tree(center_x, base_y, height, width, int(tiers))
        )

    @staticmethod
    def _validate_fir_tree(tiers, height, width):
        from numbers import Integral

        if isinstance(tiers, bool) or not isinstance(tiers, Integral) or tiers < 1:
            raise ValueError("fir_tree tiers must be an integer of at least 1")
        for name, value in (("height", height), ("width", width)):
            if isinstance(value, Real) and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"fir_tree {name} must be finite and positive")

    @staticmethod
    def _single_fir_tree(center_x, base_y, height, width):
        return build(
            "triangle",
            (center_x, base_y - height),
            (center_x - width / 2, base_y),
            (center_x + width / 2, base_y),
        )

    @staticmethod
    def _tiered_fir_tree(center_x, base_y, height, width, tiers):
        boughs = [
            SDFNamespace._fir_bough(center_x, base_y, height, width, index)
            for index in range(tiers)
        ]
        boughs.append(
            build(
                "rect",
                (center_x, base_y - height * 4 / 100),
                (width * 75 / 1000, height * 15 / 100),
                corner_radius=width * 2 / 100,
            )
        )
        return boughs

    @staticmethod
    def _fir_bough(center_x, base_y, height, width, index):
        top_y = base_y - height * (1.0 - 18 / 100 * index)
        bottom_y = base_y - height * (64 / 100 - 18 / 100 * index)
        half_width = width * (19 / 100 + 105 / 1000 * index)
        return build(
            "triangle",
            (center_x, top_y),
            (center_x - half_width, bottom_y),
            (center_x + half_width, bottom_y),
        )

    def isometric_cube(self, offset, cell_width, cell_height, stroke_width=2.0):
        """Build a wireframe isometric cube from differentiable segments.

        ``offset`` is the front lower vertex. ``cell_width`` and
        ``cell_height`` are the horizontal and vertical grid steps used by the
        diagram. All geometry values and ``stroke_width`` accept lazy
        expressions or trainable parameters.
        """
        for name, value in (
            ("cell_width", cell_width),
            ("cell_height", cell_height),
            ("stroke_width", stroke_width),
        ):
            if isinstance(value, Real) and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"isometric_cube {name} must be finite and positive")

        x, y = map(_lazy_scalar, offset)
        cell_width = _lazy_scalar(cell_width)
        cell_height = _lazy_scalar(cell_height)
        radius = _lazy_scalar(stroke_width) / 2
        points = (
            (x, y),
            (x, y + cell_height),
            (x + cell_width, y + 0.5 * cell_height),
            (x + cell_width, y + 3 / 2 * cell_height),
            (x - cell_width, y + 0.5 * cell_height),
            (x - cell_width, y + 3 / 2 * cell_height),
            (x, y + 2 * cell_height),
        )
        edge_indices = (
            (0, 1),
            (1, 3),
            (0, 2),
            (2, 3),
            (1, 5),
            (0, 4),
            (4, 5),
            (5, 6),
            (3, 6),
        )
        return self.union_all(
            build("capsule", points[first], points[second], radius)
            for first, second in edge_indices
        )

    def arrow(self, start, end, width=3.0, head_length=13.0, head_width=None):
        """Build a round shaft and triangular head between two canvas points.

        Widths and lengths are positive canvas measurements and may be lazy
        expressions. Short arrows scale the head to fit. Coincident points
        produce a round mark with the shaft's diameter.
        """
        for name, value in (
            ("width", width),
            ("head_length", head_length),
            ("head_width", head_width),
        ):
            if isinstance(value, Real) and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"arrow {name} must be finite and positive")

        ax, ay = map(_lazy_scalar, start)
        bx, by = map(_lazy_scalar, end)
        radius = _lazy_scalar(width) / 2
        head_length = _lazy_scalar(head_length)
        half_head_width = (
            52 / 100 * head_length if head_width is None else _lazy_scalar(head_width) / 2
        )
        return _arrow_head(ax, ay, bx, by, radius, head_length, half_head_width)

    def polyline(self, points, radius=0.0):
        """Create a distance field from canvas points joined by line segments.

        Insert ``None`` between runs to start a new line. Zero radius yields
        unsigned distance to the centerline. A positive radius offsets that
        distance outward in canvas units to make a filled, round-ended band.
        """
        vertices = tuple(
            tuple(float(value) if isinstance(value, Real) else value for value in point)
            if point is not None
            else None
            for point in points
        )
        if isinstance(radius, Real):
            radius = float(radius)
        return build("polyline", vertices, radius)

    def svg(
        self,
        source,
        center=(0.0, 0.0),
        height=64.0,
        samples=16,
        inflate=0.0,
        oversample=2,
    ):
        """Render filled SVG geometry, including MathJax-style SVG, as an SDF.

        ``center`` places the center of the SVG viewBox in canvas coordinates;
        ``height`` sets its rendered height while preserving its aspect ratio.
        ``inflate`` offsets the resulting SDF in canvas units. Positive values
        expand the filled geometry; negative values contract it.
        ``oversample`` evaluates subpixel samples per output pixel before
        reducing them to canvas resolution (integer 1 to 4; default 2).
        The supported SVG subset includes paths, local ``defs``/``use``
        references, affine transforms, one-line text, and nonzero or even-odd
        fill rules.
        """
        return build(
            "svg",
            source=source,
            center=center,
            height=height,
            samples=samples,
            inflate=inflate,
            oversample=oversample,
        )

    def math(
        self,
        expression,
        center=(0.0, 0.0),
        height=64.0,
        font_size=48.0,
        fontset="stix",
        samples=16,
        inflate=0.75,
        oversample=2,
    ):
        r"""Typeset a TeX-like math expression with Matplotlib MathText.

        MathText supports a useful subset of TeX math, without requiring TeX,
        Node.js, or npm. Install the optional dependency with
        ``pip install 'sdf_ui[math]'``. ``height`` controls the rendered formula
        height in canvas units. Pass ``height=None`` to use ``font_size`` as the
        font size in canvas units and let each expression have its natural height.
        ``inflate`` applies a signed-distance offset in canvas units.
        ``oversample`` improves subpixel detail and accepts values from 1 to 4.
        """
        from .mathtext import render_mathtext

        return render_mathtext(
            expression,
            center,
            height,
            font_size,
            fontset,
            samples,
            inflate,
            oversample,
        )

    def text(self, *args, **kwargs):
        """Create text, including inline spans, through the user-facing API."""
        from ..text import text

        return text(*args, **kwargs)


class ColorNamespace(_OperationNamespace):
    kind = TextureKind.COLOR

    def linear_rgba(self, rgba):
        """Create a floating point layer with linear RGBA values."""
        return build("hdr_clear", rgba)

    def concentric_rings(self, center, radius, *, count: int, tint, line_width=2.0, gap=5.0):
        """Compose equally spaced colored outlines inside a circle.

        ``count`` is a fixed positive topology control. Center, radius, tint,
        line width, and gap remain render-time inputs. Choose a count that fits
        within the radius; the innermost ring is not clipped automatically.
        """
        _validate_ring_dimensions(count, line_width, gap)

        circle = SDFNamespace().circle(center, radius)
        transparent = _transparent_ink(tint)
        spacing = _lazy_scalar(line_width) + _lazy_scalar(gap)
        result = self.clear(transparent)
        for index in range(int(count)):
            ring = circle.outline(
                tint,
                transparent,
                inflate=-index * spacing,
                width=line_width,
            ).uncached()
            result = result.alpha_overlay(ring)
            if index + 1 < count:
                result = result.uncached()
        return result

    def palette(self, value):
        """Resolve a named or custom palette for reuse in color arguments."""
        from .palette import get_palette

        return get_palette(value)
