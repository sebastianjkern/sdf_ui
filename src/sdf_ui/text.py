__docformat__ = "google"

from .fonts import measure_glyph as _measure_glyph
from .fonts import measure_text as _measure_text
from .fonts import system_font as _system_font
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


def icon(value, size=64, ox=0, oy=0, path=None, **text_options):
    """Create an SDF icon from text, including bundled-font symbol glyphs."""
    return text(value, size=size, ox=ox, oy=oy, path=path, **text_options)


def badge(
    value,
    center=(0, 0),
    size=24,
    padding=(12, 7),
    radius=8,
    fg_color=(1.0, 1.0, 1.0, 1.0),
    bg_color=(0.12, 0.32, 0.82, 1.0),
    path=None,
):
    """Create a centered text badge as a composable color texture.

    ``size`` is the text size in pixels. ``padding`` is horizontal and vertical
    padding; ``radius`` is the rounded-rectangle corner radius.
    """
    from .gpu.nodes.common import build

    metrics = measure_text(value, size=size, path=path)
    width = metrics["width"] + 2 * padding[0]
    height = metrics["height"] + 2 * padding[1]
    background = build(
        "rounded_rect", center, (width, height), radius
    ).fill(bg_color)
    box = (center[0] - width / 2, center[1] - height / 2, width, height)
    label = text(value, size=size, path=path, box=box, align="center", valign="middle")
    return background.alpha_overlay(label.fill(fg_color, (0.0, 0.0, 0.0, 0.0)))


def text(
    value,
    size=64,
    ox=0,
    oy=0,
    path=None,
    samples=16,
    cache_size=128,
    line_height=1.2,
    oversample=2.5,
    min_render_size=64,
    box=None,
    align="left",
    valign="middle",
    fallback_fonts=(),
    missing_glyph="ignore",
    clip=False,
    overflow="visible",
    trim_trailing_spaces=True,
    direction=None,
    script=None,
    language=None,
    features=None,
    variations=None,
    white_space="normal",
    first_line_indent=0.0,
    max_lines=None,
    decoration=(),
    decoration_thickness=0.0,
    word_spacing=0.0,
    letter_spacing=0.0,
    vertical_metrics="hhea",
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
    width and can use ``align="left"``, ``"center"``, ``"right"``, or
    ``"justify"``. Justification expands spaces on non-final paragraph lines.
    ``overflow`` selects ``"visible"``, ``"clip"``, or ``"error"`` behavior;
    the older ``clip=True`` option is shorthand for ``overflow="clip"``.
    ``trim_trailing_spaces`` controls whether line-ending spaces are removed
    before measuring and drawing. Without a box, the
    existing ``ox`` and ``oy`` baseline placement is preserved. The box positions
    text without clipping overflow by default. ``fallback_fonts`` lists
    font paths to try for missing characters. ``missing_glyph`` can be ``"ignore"``
    (the backward-compatible default) or ``"error"``.
    """
    options = locals().copy()
    if not isinstance(value, str):
        return _text_spans(value, **_span_options(options))
    return build("text", **options)


def _span_options(options):
    options.pop("value")
    options["common_decoration"] = options.pop("decoration")
    options["common_decoration_thickness"] = options.pop("decoration_thickness")
    clip = options.pop("clip")
    if clip and options["overflow"] == "visible":
        options["overflow"] = "clip"
    return options


def span(value, **style):
    """Create an inline text span for ``text([span(...), ...])``."""
    if not isinstance(value, str):
        raise TypeError("span value must be a string")
    return {"text": value, **style}


def _text_spans(value, **common):
    from ._inline_text import render_inline_text

    return render_inline_text(value, common, text)


def measure_text(
    value,
    size=64,
    path=None,
    *,
    width=None,
    height=None,
    line_height=1.2,
    fallback_fonts=(),
    missing_glyph="ignore",
    direction=None,
    script=None,
    language=None,
    features=None,
    variations=None,
    trim_trailing_spaces=True,
    white_space="normal",
    word_spacing=0.0,
    letter_spacing=0.0,
    first_line_indent=0.0,
    max_lines=None,
    vertical_metrics="hhea",
):
    """Measure text with the same fallback-font and wrapping rules as ``text``.

    Returns maximum line width, total block height, laid-out lines, each line's
    advance width in canvas pixels, and whether the result exceeds the supplied
    ``width`` or ``height``. Supply ``width`` to wrap the text.
    """
    return _measure_text(
        value,
        size=size,
        path=path,
        fallbacks=fallback_fonts,
        box_width=width,
        box_height=height,
        line_height=line_height,
        missing_glyph=missing_glyph,
        direction=direction,
        script=script,
        language=language,
        features=features,
        variations=variations,
        trim_trailing_spaces=trim_trailing_spaces,
        white_space=white_space,
        word_spacing=word_spacing,
        letter_spacing=letter_spacing,
        first_line_indent=first_line_indent,
        max_lines=max_lines,
        overflow="ellipsis" if max_lines is not None else "visible",
        vertical_metrics=vertical_metrics,
    )


def measure_glyph(char, size=64, path=None, *, fallback_fonts=(), variations=None):
    """Measure one shaped grapheme without creating a render node."""
    return _measure_glyph(
        char, size=size, path=path, fallbacks=fallback_fonts, variations=variations
    )


def system_font(family="sans-serif", style="Regular"):
    """Return a system font file path for use as ``text(path=...)``."""
    return _system_font(family, style)
