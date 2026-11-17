"""Dynamic namespaces that expose plugin-backed factory functions."""

__docformat__ = "google"

from ..gpu.nodes.base import TextureKind
from ..gpu.nodes.common import build
from ._suggestions import unknown_name_message


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
        height in canvas units; ``font_size`` controls MathText's source size.
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
