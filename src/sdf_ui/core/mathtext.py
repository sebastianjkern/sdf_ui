"""MathText-to-SVG adapter for the SDF SVG primitive."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from io import BytesIO


def render_mathtext(
    expression, center, height, font_size, fontset, samples, inflate, oversample
):
    """Typeset a MathText expression and return it as an SVG-backed SDF."""
    math_expression = _validated_math_expression(expression, height, font_size)
    root = _render_math_svg(math_expression, font_size, fontset)
    _remove_math_background(root)
    return _build_math_sdf(root, center, height, samples, inflate, oversample)


def _validated_math_expression(expression, height, font_size):
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("math expression must be a non-empty string")
    sizes = (font_size,) if height is None else (height, font_size)
    if not all(math.isfinite(value) and value > 0 for value in sizes):
        raise ValueError("height and font_size must be finite and positive")
    stripped = expression.strip()
    return stripped if stripped.startswith("$") and stripped.endswith("$") else f"${stripped}$"


def _render_math_svg(math_expression, font_size, fontset):
    try:
        from matplotlib import rc_context
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import math_to_image
    except ImportError as exc:
        raise ImportError(
            "sdf.math requires Matplotlib MathText; install it with "
            "`pip install 'sdf_ui[math]'`."
        ) from exc

    output = BytesIO()
    with rc_context({"mathtext.fontset": fontset, "svg.fonttype": "path"}):
        math_to_image(
            math_expression,
            output,
            prop=FontProperties(size=font_size),
            dpi=72,
            format="svg",
            color="black",
        )

    return ET.fromstring(output.getvalue())


def _remove_math_background(root):
    # math_to_image adds an opaque page rectangle. The SVG primitive renders
    # filled paths as geometry, so remove the page while retaining glyph paths.
    for parent in root.iter():
        for child in list(parent):
            if child.attrib.get("id") == "patch_1":
                parent.remove(child)


def _build_math_sdf(root, center, height, samples, inflate, oversample):

    from ..gpu.nodes.common import build

    return build(
        "svg",
        source=ET.tostring(root, encoding="unicode"),
        center=center,
        height=float(root.attrib["viewBox"].split()[3]) if height is None else height,
        samples=samples,
        inflate=inflate,
        oversample=oversample,
    )
