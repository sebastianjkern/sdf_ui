__docformat__ = "google"

from .fonts import measure_text as _measure_text
from .gpu.nodes.common import build


def glyph(char, scale, ox, oy, path=None, samples=16):
    """
    Generate the signed distance field (SDF) for a glyph.

    Parameters:
    - char (str): The character for which to generate the SDF.
    - scale (float): Scaling factor for the glyph.
    - ox, oy (float): Offset of the glyph.
    - path (str): Path to the font file.
    - samples (int): Number of line segments used to approximate font curves.

    Returns:
        SDFTexture: A render node producing the filled glyph SDF.
    """
    if len(char) != 1:
        raise ValueError("glyph expects exactly one character")
    return build(
        "glyph",
        char=char,
        scale=scale,
        ox=ox,
        oy=oy,
        path=path,
        samples=samples,
    )


def text(
    value,
    size=64,
    ox=0,
    oy=0,
    path=None,
    samples=16,
    cache_size=128,
    line_height=1.2,
    oversample=2.0,
    min_render_size=64,
    box=None,
    align="left",
    valign="middle",
    fallback_fonts=(),
    missing_glyph="ignore",
    direction=None,
    script=None,
    language=None,
    features=None,
    variations=None,
):
    """
    Generate a signed distance field (SDF) for a text run.

    Glyph SDF patches are cached by font, glyph ID, curve sampling, and
    requested pixel size. ``cache_size`` sets the minimum cached pixels-per-em
    value, ``oversample`` raises the cache resolution for larger text, and
    ``min_render_size`` keeps small text from losing too much SDF distance when
    resized down. Text is shaped with HarfBuzz, including OpenType substitutions,
    positioning, and bidirectional runs. ``direction``, ``script``, and
    ``language`` override HarfBuzz's inferred run properties. ``features`` is
    a mapping such as ``{"liga": False, "kern": True}``; ``variations`` maps
    variable-font axis tags to values. ``box`` is ``(x, y, width, height)`` in
    lower-left-origin canvas coordinates. When supplied, text wraps to the box
    width and can be aligned with ``align`` and ``valign``. Without a box, the
    existing ``ox`` and ``oy`` baseline placement is preserved. The box positions text but does
    not clip lines that overflow its height. ``fallback_fonts`` lists font
    paths to try for missing characters. ``missing_glyph`` can be ``"ignore"``
    (the backward-compatible default) or ``"error"``.
    """
    return build(
        "text",
        value=value,
        size=size,
        ox=ox,
        oy=oy,
        path=path,
        samples=samples,
        cache_size=cache_size,
        line_height=line_height,
        oversample=oversample,
        min_render_size=min_render_size,
        box=box,
        align=align,
        valign=valign,
        fallback_fonts=fallback_fonts,
        missing_glyph=missing_glyph,
        direction=direction,
        script=script,
        language=language,
        features=features,
        variations=variations,
    )


def measure_text(
    value,
    size=64,
    path=None,
    *,
    width=None,
    line_height=1.2,
    fallback_fonts=(),
    missing_glyph="ignore",
    direction=None,
    script=None,
    language=None,
    features=None,
    variations=None,
):
    """Measure text with the same fallback-font and wrapping rules as ``text``.

    Returns maximum line width, total block height, laid-out lines, and each
    line's advance width in canvas pixels. Supply ``width`` to wrap the text.
    """
    return _measure_text(
        value,
        size=size,
        path=path,
        fallbacks=fallback_fonts,
        box_width=width,
        line_height=line_height,
        missing_glyph=missing_glyph,
        direction=direction,
        script=script,
        language=language,
        features=features,
        variations=variations,
    )
