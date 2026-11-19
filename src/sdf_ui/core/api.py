"""Dynamic namespaces that expose plugin-backed factory functions."""

__docformat__ = "google"

import math
from numbers import Real

from ..gpu.nodes.base import TextureKind
from ..gpu.nodes.common import build
from ._suggestions import unknown_name_message
from .expressions import (
    _lazy_scalar,
    _scalar_maximum,
    _scalar_minimum,
    _select_positive,
)


class _OperationNamespace:
    kind = None

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
            0.52 * head_length if head_width is None else _lazy_scalar(head_width) / 2
        )
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        # At subpixel lengths use a capsule and keep a valid head inside it.
        # This also keeps a graph compiled at positive length valid at zero.
        direction = length_sq - 1e-6
        if isinstance(direction, Real) and direction <= 0:
            return build("capsule", (ax, ay), (bx, by), radius)
        length = _scalar_maximum(length_sq, 1e-6) ** 0.5
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

    def polyline(self, points, radius=0.0):
        """Create a distance field from canvas points joined by line segments.

        Insert ``None`` between runs to start a new line. Zero radius yields
        unsigned distance to the centerline. A positive radius offsets that
        distance outward in canvas units to make a filled, round-ended band.
        """
        vertices = tuple(
            tuple(float(value) if isinstance(value, Real) else value for value in point)
            if point is not None else None
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

    def palette(self, value):
        """Resolve a named or custom palette for reuse in color arguments."""
        from .palette import get_palette

        return get_palette(value)
