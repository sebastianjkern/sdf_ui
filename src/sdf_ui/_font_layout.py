"""Paragraph layout, wrapping, and text overflow handling."""

import re
from math import isfinite
from numbers import Real

from uniseg.linebreak import LB, line_break, line_break_units


def layout_text(
    value,
    paths,
    size,
    *,
    box_width=None,
    missing_glyph="ignore",
    direction=None,
    script=None,
    language=None,
    features=None,
    variations=None,
    word_spacing=0.0,
    letter_spacing=0.0,
    white_space="normal",
    trim_trailing_spaces=None,
    first_line_indent=0.0,
    max_lines=None,
    overflow="visible",
):
    """Return wrapped lines as positioned OpenType glyph IDs and pixel advances."""
    from .fonts import _shape_with_options

    trim_trailing_spaces = _validate_layout_options(
        value,
        missing_glyph,
        direction,
        white_space,
        trim_trailing_spaces,
        word_spacing,
        letter_spacing,
        first_line_indent,
        max_lines,
        overflow,
    )
    options = _shape_options(direction, script, language, features, variations,
                             word_spacing, letter_spacing)

    def shape(text):
        return _shape_with_options(text, paths, size, missing_glyph, options)

    lines = _layout_paragraphs(
        value, shape, white_space, trim_trailing_spaces, box_width, first_line_indent
    )
    return _truncate_lines(
        lines, max_lines, overflow, shape, paths, box_width, first_line_indent
    )


def _validate_layout_options(
    value,
    missing_glyph,
    direction,
    white_space,
    trim_trailing_spaces,
    word_spacing,
    letter_spacing,
    first_line_indent,
    max_lines,
    overflow,
):
    _validate_text_modes(value, missing_glyph, direction, white_space)
    _validate_spacing_values(word_spacing, letter_spacing, first_line_indent)
    _validate_overflow(max_lines, overflow)
    return _resolve_trailing_space_mode(trim_trailing_spaces, white_space)


def _validate_text_modes(value, missing_glyph, direction, white_space):
    if not isinstance(value, str):
        raise TypeError("text value must be a string")
    if missing_glyph not in {"ignore", "error"}:
        raise ValueError("missing_glyph must be 'ignore' or 'error'")
    if direction not in {None, "ltr", "rtl"}:
        raise ValueError("text direction must be None, 'ltr', or 'rtl'")
    if white_space not in {"normal", "pre", "pre-wrap"}:
        raise ValueError("white_space must be 'normal', 'pre', or 'pre-wrap'")


def _resolve_trailing_space_mode(trim_trailing_spaces, white_space):
    if trim_trailing_spaces is None:
        return white_space == "normal"
    if not isinstance(trim_trailing_spaces, bool):
        raise TypeError("trim_trailing_spaces must be a bool or None")
    return trim_trailing_spaces


def _validate_spacing_values(word_spacing, letter_spacing, first_line_indent):
    for name, value_ in (
        ("word_spacing", word_spacing),
        ("letter_spacing", letter_spacing),
        ("first_line_indent", first_line_indent),
    ):
        if (
            isinstance(value_, bool)
            or not isinstance(value_, Real)
            or not isfinite(value_)
        ):
            raise ValueError(f"{name} must be a finite number")


def _validate_overflow(max_lines, overflow):
    if max_lines is not None and (
        isinstance(max_lines, bool) or not isinstance(max_lines, int) or max_lines < 1
    ):
        raise ValueError("max_lines must be a positive integer")
    if overflow not in {"visible", "clip", "error", "ellipsis"}:
        raise ValueError("unsupported text overflow mode")
    if overflow == "ellipsis" and max_lines is None:
        raise ValueError("ellipsis overflow requires max_lines")


def _shape_options(direction, script, language, features, variations, word_spacing, letter_spacing):
    return {
        "direction": direction,
        "script": script,
        "language": language,
        "features": features,
        "variations": variations,
        "word_spacing": float(word_spacing),
        "letter_spacing": float(letter_spacing),
    }


def _layout_paragraphs(value, shape, white_space, trim_trailing_spaces, box_width, indent):
    lines = []
    paragraphs = re.split(r"\r\n|[\r\n\v\f\x85\u2028\u2029]", value)
    for paragraph in paragraphs:
        paragraph = _normalize_paragraph(paragraph, white_space, trim_trailing_spaces)
        if not paragraph:
            lines.append((*shape(""), True, False, True))
            continue
        wrapped = _wrap_or_shape(
            paragraph, shape, box_width, white_space, trim_trailing_spaces, indent
        )
        lines.extend(
            (*line, index == len(wrapped) - 1, False, index == 0)
            for index, line in enumerate(wrapped)
        )
    return lines


def _normalize_paragraph(paragraph, white_space, trim_trailing_spaces):
    if white_space != "normal":
        return _expand_tabs(paragraph)
    if trim_trailing_spaces:
        return re.sub(r"[ \t]+", " ", paragraph).strip(" \t")
    trailing = re.search(r"[ \t]+$", paragraph)
    trailing_spaces = trailing.group() if trailing else ""
    return re.sub(r"[ \t]+", " ", paragraph.rstrip(" \t")) + trailing_spaces


def _wrap_or_shape(paragraph, shape, box_width, white_space, trim_trailing_spaces, indent):
    if box_width is None or white_space == "pre":
        text = paragraph.rstrip(" \t") if trim_trailing_spaces else paragraph
        return [shape(text)]
    return _wrap_paragraph(
        paragraph,
        shape,
        box_width,
        trim_trailing_spaces=trim_trailing_spaces,
        first_line_indent=indent,
    )


def _truncate_lines(lines, max_lines, overflow, shape, paths, box_width, indent):
    if max_lines is None or len(lines) <= max_lines:
        return lines
    lines = lines[:max_lines]
    placements, width, text, _last, _truncated, first = lines[-1]
    if overflow != "ellipsis":
        lines[-1] = (placements, width, text, True, True, first)
        return lines
    lines[-1] = _ellipsize(text, first, shape, paths, box_width, indent)
    return lines


def _ellipsize(text, first, shape, paths, box_width, indent):
    from ._font_graphemes import graphemes as _graphemes
    from ._font_resources import _cmap

    marker = "…" if any(0x2026 in _cmap(path) for path in paths) else "..."
    clusters = list(_graphemes(text.rstrip(" \t")))
    available = None if box_width is None else box_width - (indent if first else 0.0)
    while True:
        candidate = "".join(clusters) + marker
        ellipsized = shape(candidate)
        if available is None or ellipsized[1] <= available or not clusters:
            return (*ellipsized, True, True, first)
        clusters.pop()


def _expand_tabs(value, tab_size=4):
    expanded = []
    column = 0
    for char in value:
        if char == "\t":
            count = tab_size - column % tab_size
            expanded.append(" " * count)
            column += count
        else:
            expanded.append(char)
            column = 0 if char in "\r\n" else column + 1
    return "".join(expanded)


def _wrap_paragraph(
    paragraph,
    shape,
    box_width,
    *,
    trim_trailing_spaces=True,
    first_line_indent=0.0,
):
    lines = []
    line = ""
    for unit in line_break_units(paragraph):
        line_width = box_width - first_line_indent if not lines else box_width
        line = _process_wrap_unit(
            unit, line, lines, shape, line_width, box_width,
            trim_trailing_spaces, first_line_indent,
        )

    if line or not lines:
        text = line.rstrip() if trim_trailing_spaces else line
        lines.append(shape(text))
    return lines


def _process_wrap_unit(unit, line, lines, shape, line_width, box_width, trim, indent):
    candidate = line + unit
    measured = candidate.rstrip() if trim else candidate
    if shape(measured)[1] <= line_width:
        return candidate
    _commit_line(lines, line, shape, trim)
    unit = unit.lstrip() if trim else unit
    if not unit:
        return ""
    next_width = box_width - indent if not lines else box_width
    complete, remainder = _wrap_unit(unit, shape, next_width, box_width, trim)
    lines.extend(complete)
    return remainder


def _commit_line(lines, line, shape, trim_trailing_spaces):
    text = line.rstrip() if trim_trailing_spaces else line
    if text:
        lines.append(shape(text))


def _wrap_unit(unit, shape, line_width, box_width, trim_trailing_spaces):
    measured = unit.rstrip() if trim_trailing_spaces else unit
    if shape(measured)[1] <= line_width:
        return [], unit
    chunks = _wrap_long_word(unit, shape, line_width, box_width)
    return [shape(chunk) for chunk in chunks[:-1]], chunks[-1]


def _wrap_long_word(word, shape, box_width, continuation_width=None):
    from ._font_graphemes import graphemes as _graphemes

    if any(line_break(char) in {LB.GL, LB.WJ, LB.ZWJ} for char in word):
        return [word]
    chunks = []
    chunk = ""
    for cluster in _graphemes(word):
        candidate = chunk + cluster
        width = box_width if not chunks else continuation_width or box_width
        if chunk and shape(candidate)[1] > width:
            chunks.append(chunk)
            chunk = cluster
        else:
            chunk = candidate
    if chunk:
        chunks.append(chunk)
    return chunks
