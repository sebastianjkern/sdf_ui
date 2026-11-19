__docformat__ = "google"

from math import isfinite
from numbers import Real

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
    if not isinstance(value, str):
        return _text_spans(
            value,
            size=size,
            ox=ox,
            oy=oy,
            path=path,
            samples=samples,
            cache_size=cache_size,
            line_height=line_height,
            oversample=oversample,
            min_render_size=min_render_size,
            fallback_fonts=fallback_fonts,
            missing_glyph=missing_glyph,
            direction=direction,
            script=script,
            language=language,
            features=features,
            variations=variations,
            trim_trailing_spaces=trim_trailing_spaces,
            common_decoration=decoration,
            common_decoration_thickness=decoration_thickness,
            word_spacing=word_spacing,
            letter_spacing=letter_spacing,
            vertical_metrics=vertical_metrics,
            box=box,
            align=align,
            valign=valign,
            overflow="clip" if clip and overflow == "visible" else overflow,
            white_space=white_space,
            first_line_indent=first_line_indent,
            max_lines=max_lines,
        )
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
        clip=clip,
        overflow=overflow,
        trim_trailing_spaces=trim_trailing_spaces,
        direction=direction,
        script=script,
        language=language,
        features=features,
        variations=variations,
        white_space=white_space,
        first_line_indent=first_line_indent,
        max_lines=max_lines,
        decoration=decoration,
        decoration_thickness=decoration_thickness,
        word_spacing=word_spacing,
        letter_spacing=letter_spacing,
        vertical_metrics=vertical_metrics,
    )


def span(value, **style):
    """Create an inline text span for ``text([span(...), ...])``."""
    if not isinstance(value, str):
        raise TypeError("span value must be a string")
    return {"text": value, **style}


def _text_spans(value, **common):
    import re

    from uniseg.linebreak import line_break_units

    from .fonts import _graphemes
    from .fonts import measure_text as measure_shaped_text

    if not isinstance(value, (list, tuple)):
        raise TypeError("inline text value must be a sequence of span() values")
    box = common["box"]
    if box is None and common["overflow"] != "visible":
        raise ValueError("inline text overflow modes require a box")
    if common["overflow"] not in {"visible", "clip", "error", "ellipsis"}:
        raise ValueError("unsupported inline text overflow mode")
    if common["overflow"] == "ellipsis" and common["max_lines"] is None:
        raise ValueError("ellipsis overflow requires max_lines")
    if box is None and common["align"] != "left":
        raise ValueError("inline text alignment requires a box")
    if common["align"] not in {"left", "center", "right", "justify"}:
        raise ValueError("unsupported inline text alignment")
    if common["valign"] not in {"top", "middle", "bottom"}:
        raise ValueError("unsupported inline text vertical alignment")
    if common["max_lines"] is not None and (
        isinstance(common["max_lines"], bool)
        or not isinstance(common["max_lines"], int)
        or common["max_lines"] < 1
    ):
        raise ValueError("max_lines must be a positive integer")
    if box is not None:
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            raise TypeError("text box must be (x, y, width, height)")
        if any(
            isinstance(item, bool) or not isinstance(item, Real) or not isfinite(item)
            for item in box
        ):
            raise ValueError("text box values must be finite numbers")
        x, y, box_width, box_height = map(float, box)
        if box_width <= 0 or box_height <= 0:
            raise ValueError("text box width and height must be positive")
    else:
        x, y, box_width, box_height = common["ox"], common["oy"], None, None
    indent = common["first_line_indent"]
    if isinstance(indent, bool) or not isinstance(indent, Real) or not isfinite(indent):
        raise ValueError("first_line_indent must be a finite number")

    allowed = {
        "size",
        "path",
        "fallback_fonts",
        "missing_glyph",
        "direction",
        "script",
        "language",
        "features",
        "variations",
        "decoration",
        "decoration_thickness",
        "word_spacing",
        "letter_spacing",
    }
    base_options = {
        "samples": common["samples"],
        "cache_size": common["cache_size"],
        "line_height": common["line_height"],
        "oversample": common["oversample"],
        "min_render_size": common["min_render_size"],
        "missing_glyph": common["missing_glyph"],
        "direction": common["direction"],
        "script": common["script"],
        "language": common["language"],
        "features": common["features"],
        "variations": common["variations"],
        "trim_trailing_spaces": False,
        "decoration": common["common_decoration"],
        "decoration_thickness": common["common_decoration_thickness"],
        "word_spacing": common["word_spacing"],
        "letter_spacing": common["letter_spacing"],
        "vertical_metrics": common["vertical_metrics"],
    }
    default_style = {
        "size": common["size"],
        "path": common["path"],
        "fallback_fonts": common["fallback_fonts"],
        **base_options,
    }
    lines = [{"segments": [], "width": 0.0, "first": True, "last": False}]
    line_index = 0

    def measure(value_, style):
        return measure_shaped_text(
            value_,
            size=style["size"],
            path=style["path"],
            fallbacks=style["fallback_fonts"],
            line_height=common["line_height"],
            missing_glyph=style["missing_glyph"],
            direction=style["direction"],
            script=style["script"],
            language=style["language"],
            features=style["features"],
            variations=style["variations"],
            trim_trailing_spaces=False,
            word_spacing=style["word_spacing"],
            letter_spacing=style["letter_spacing"],
        )

    def add_segment(value_, style):
        if not value_:
            return
        line = lines[-1]
        if line["segments"] and line["segments"][-1][1] == style:
            previous = line["segments"].pop()
            value_ = previous[0] + value_
        metrics = measure(value_, style)
        line["segments"].append((value_, style, metrics))
        line["width"] = sum(part[2]["width"] for part in line["segments"])

    def next_line(*, paragraph_first=False, paragraph_last=False):
        nonlocal line_index
        line = lines[-1]
        if common["trim_trailing_spaces"] and common["white_space"] != "pre":
            while line["segments"] and line["segments"][-1][0].endswith((" ", "\t")):
                content, style, _metrics = line["segments"].pop()
                content = content.rstrip(" \t")
                if content:
                    add_segment(content, style)
            line["width"] = sum(part[2]["width"] for part in line["segments"])
        line["last"] = paragraph_last
        line_index += 1
        lines.append(
            {"segments": [], "width": 0.0, "first": paragraph_first, "last": False}
        )

    def available_width():
        if box_width is None:
            return None
        return box_width - (common["first_line_indent"] if lines[-1]["first"] else 0.0)

    def append_unit(unit, style):
        if common["white_space"] == "normal":
            unit = re.sub(r"[ \t]+", " ", unit)
        if not unit:
            return
        limit = available_width()
        width = measure(unit, style)["width"]
        line = lines[-1]
        if (
            limit is not None
            and common["white_space"] != "pre"
            and line["width"]
            and line["width"] + width > limit
        ):
            if common["white_space"] == "normal" and unit.isspace():
                return
            next_line()
            line = lines[-1]
            limit = available_width()
        if limit is not None and common["white_space"] != "pre" and width > limit:
            chunk = ""
            for grapheme in _graphemes(unit):
                candidate = chunk + grapheme
                if chunk and measure(candidate, style)["width"] > (
                    available_width() or limit
                ):
                    add_segment(chunk, style)
                    next_line()
                    chunk = grapheme
                else:
                    chunk = candidate
            add_segment(chunk, style)
        else:
            add_segment(unit, style)

    for item in value:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            content, style_override = item
            item = {"text": content, **style_override}
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise TypeError(
                "each inline span must be span(text, **style) or (text, style)"
            )
        override = {key: item[key] for key in item if key != "text"}
        unknown = set(override) - allowed
        if unknown:
            raise TypeError(
                f"unsupported inline text styles: {', '.join(sorted(unknown))}"
            )
        style = {**default_style, **override}
        paragraphs = item["text"].replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for paragraph_index, paragraph in enumerate(paragraphs):
            if paragraph:
                if common["white_space"] == "pre":
                    append_unit(paragraph, style)
                else:
                    for unit in line_break_units(paragraph):
                        append_unit(unit, style)
            if paragraph_index < len(paragraphs) - 1:
                next_line(paragraph_first=True, paragraph_last=True)

    lines[-1]["last"] = True
    if not lines[-1]["segments"] and len(lines) > 1:
        lines.pop()
    max_lines = common["max_lines"]
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1]["last"] = True
        lines[-1]["truncated"] = True
        if common["overflow"] == "ellipsis":
            ellipsis_style = (
                lines[-1]["segments"][-1][1] if lines[-1]["segments"] else default_style
            )
            limit = (
                box_width - (common["first_line_indent"] if lines[-1]["first"] else 0.0)
                if box_width is not None
                else None
            )
            while (
                limit is not None
                and lines[-1]["width"] + measure("…", ellipsis_style)["width"] > limit
                and lines[-1]["segments"]
            ):
                content, style, _metrics = lines[-1]["segments"].pop()
                clusters = list(_graphemes(content))
                if clusters:
                    clusters.pop()
                shortened = "".join(clusters)
                if shortened:
                    add_segment(shortened, style)
            add_segment("…", ellipsis_style)

    largest_run_size = max(
        (
            float(style["size"])
            for line in lines
            for _content, style, _metrics in line["segments"]
        ),
        default=float(common["size"]),
    )
    line_advance = largest_run_size * float(common["line_height"])
    line_ascents = [
        max(
            (part[2]["ascent"] for part in line["segments"]),
            default=float(common["size"]) * 0.8,
        )
        for line in lines
    ]
    line_descents = [
        min(
            (part[2]["descent"] for part in line["segments"]),
            default=-float(common["size"]) * 0.2,
        )
        for line in lines
    ]
    ascent = max(line_ascents, default=float(common["size"]) * 0.8)
    descent = min(line_descents, default=-float(common["size"]) * 0.2)
    block_height = ascent - descent + max(len(lines) - 1, 0) * line_advance
    measured_overflow = box is not None and (
        block_height > box_height
        or any(
            line["width"] + (common["first_line_indent"] if line["first"] else 0.0)
            > box_width
            for line in lines
        )
    )
    if common["overflow"] == "error" and (
        measured_overflow or any(line.get("truncated") for line in lines)
    ):
        raise ValueError("text layout exceeds the configured box")

    if box is None:
        first_baseline = y
    elif common["valign"] == "top":
        first_baseline = y + box_height - ascent
    elif common["valign"] == "bottom":
        first_baseline = y - descent + max(len(lines) - 1, 0) * line_advance
    else:
        first_baseline = (
            y
            + (box_height - block_height) / 2
            - descent
            + max(len(lines) - 1, 0) * line_advance
        )

    nodes = []
    for index, line in enumerate(lines):
        width = line["width"]
        indent = common["first_line_indent"] if line["first"] else 0.0
        if common["align"] == "center" and box is not None:
            start_x = x + (box_width - width) / 2
        elif common["align"] == "right" and box is not None:
            start_x = x + box_width - width
        else:
            start_x = x + indent
        justify_spaces = sum(
            content.count(" ") for content, _style, _m in line["segments"]
        )
        target_width = box_width - indent if box_width is not None else width
        extra_space = (
            max(0.0, target_width - width) / justify_spaces
            if common["align"] == "justify" and not line["last"] and justify_spaces
            else 0.0
        )
        cursor = start_x
        baseline = first_baseline - index * line_advance
        for content, style, metrics in line["segments"]:
            pieces = re.split(r"( +)", content) if extra_space else (content,)
            for piece in pieces:
                piece_metrics = metrics if piece == content else measure(piece, style)
                if piece.strip():
                    options = {key: style[key] for key in base_options}
                    nodes.append(
                        text(
                            piece,
                            size=style["size"],
                            ox=cursor,
                            oy=baseline,
                            path=style["path"],
                            fallback_fonts=style["fallback_fonts"],
                            **options,
                        )
                    )
                cursor += piece_metrics["width"]
                if extra_space and piece.isspace():
                    cursor += extra_space * len(piece)
    if not nodes:
        return text(
            "",
            size=common["size"],
            ox=common["ox"],
            oy=common["oy"],
            path=common["path"],
        )
    result = nodes[0]
    for node in nodes[1:]:
        result = result.union(node)
    if box is not None and common["overflow"] == "clip":
        from .core.api import SDFNamespace

        clip_shape = SDFNamespace().rect(
            (x + box_width / 2, y + box_height / 2), (box_width, box_height)
        )
        result = result.intersection(clip_shape)
    return result


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
