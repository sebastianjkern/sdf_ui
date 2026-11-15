"""MathText-to-SVG adapter for the SDF SVG primitive."""

from __future__ import annotations

from io import BytesIO
import math
import xml.etree.ElementTree as ET


def render_mathtext(
    expression, center, height, font_size, fontset, samples, inflate, oversample
):
    """Typeset a MathText expression and return it as an SVG-backed SDF."""
    try:
        from matplotlib import rc_context
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import math_to_image
    except ImportError as exc:
        raise ImportError(
            "sdf.math requires Matplotlib MathText; install it with "
            "`pip install 'sdf_ui[math]'`."
        ) from exc

    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("math expression must be a non-empty string")
    if not all(math.isfinite(value) and value > 0 for value in (height, font_size)):
        raise ValueError("height and font_size must be finite and positive")

    math_expression = expression.strip()
    if not (math_expression.startswith("$") and math_expression.endswith("$")):
        math_expression = f"${math_expression}$"

    output = BytesIO()
    with rc_context({"mathtext.fontset": fontset}):
        math_to_image(
            math_expression,
            output,
            prop=FontProperties(size=font_size),
            dpi=72,
            format="svg",
            color="black",
        )

    root = ET.fromstring(output.getvalue())
    # math_to_image adds an opaque page rectangle. The SVG primitive renders
    # filled paths as geometry, so remove the page while retaining glyph paths.
    for parent in root.iter():
        for child in list(parent):
            if child.attrib.get("id") == "patch_1":
                parent.remove(child)

    from ..gpu.nodes.common import build

    return build(
        "svg",
        source=ET.tostring(root, encoding="unicode"),
        center=center,
        height=height,
        samples=samples,
        inflate=inflate,
        oversample=oversample,
    )
