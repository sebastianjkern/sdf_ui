"""Font discovery, OpenType shaping, and shared text layout."""

from math import isfinite
from numbers import Real

import uharfbuzz as hb

from ._font_bidi import bidi_runs as _bidi_runs
from ._font_bidi import script_runs as _script_runs
from ._font_graphemes import graphemes as _graphemes
from ._font_graphemes import is_variation_selector as _is_variation_selector
from ._font_layout import layout_text
from ._font_resources import (
    _cmap,
    _hb_font,
    font_metrics,
    font_paths,
    glyph_id_for_char,
    glyph_metrics,
    layout_font_paths,
    text_vertical_metrics,
)
from ._font_resources import (
    default_font_path as default_font_path,
)
from ._font_resources import (
    system_font as system_font,
)
from ._font_resources import (
    text_decoration_metrics as text_decoration_metrics,
)


def _font_for_cluster(cluster, paths, missing_glyph):
    codepoints = [
        ord(char)
        for char in cluster
        if char != "\u200d" and not _is_variation_selector(char)
    ]
    for path in paths:
        cmap = _cmap(path)
        if all(codepoint in cmap for codepoint in codepoints):
            infos, _positions = _shape_run(cluster, path)
            if all(info.codepoint != 0 for info in infos):
                return path
    if missing_glyph == "error":
        missing = next(
            (c for c in cluster if all(ord(c) not in _cmap(path) for path in paths)),
            cluster[0],
        )
        raise ValueError(
            f"No configured font can shape U+{ord(missing):04X} ({missing!r})"
        )
    return None


def _shape_run(
    value,
    path,
    *,
    direction=None,
    script=None,
    language=None,
    features=None,
    variations=None,
):
    variation_items = tuple(sorted((variations or {}).items()))
    font = _hb_font(path, variation_items)
    buffer = hb.Buffer()
    buffer.add_str(value)
    if direction:
        buffer.direction = direction
    if script:
        buffer.script = script
    if language:
        buffer.language = language
    buffer.guess_segment_properties()
    hb.shape(font, buffer, features or {})
    return buffer.glyph_infos, buffer.glyph_positions


def _shape_line(
    value,
    paths,
    size,
    missing_glyph,
    *,
    direction=None,
    script=None,
    language=None,
    features=None,
    variations=None,
    word_spacing=0.0,
    letter_spacing=0.0,
):
    visual_runs = _bidi_runs(value, direction)
    placements = []
    pen_x = 0.0
    low_x = 0.0
    run_index = 0
    for visual_run, run_direction in visual_runs:
        script_runs = _script_runs(visual_run, script)
        if run_direction == "rtl":
            script_runs.reverse()
        for script_run, run_script in script_runs:
            font_runs = _font_runs(script_run, paths, missing_glyph)
            if run_direction == "rtl":
                font_runs.reverse()
            new_placements, pen_x, low_x, run_index = _shape_font_runs(
                font_runs,
                run_direction,
                run_script,
                language,
                features,
                variations,
                size,
                pen_x,
                low_x,
                run_index,
            )
            placements.extend(new_placements)

    return _finish_shaped_line(
        placements, pen_x, low_x, value, word_spacing, letter_spacing
    )


def _font_runs(script_run, paths, missing_glyph):
    runs = []
    for cluster in _graphemes(script_run):
        path = _font_for_cluster(cluster, paths, missing_glyph)
        if path is None:
            continue
        if runs and runs[-1][0] == path:
            runs[-1] = (path, runs[-1][1] + cluster)
        else:
            runs.append((path, cluster))
    return runs


def _shape_font_runs(
    font_runs, direction, script, language, features, variations, size,
    pen_x, low_x, run_index,
):
    placements = []
    for path, run in font_runs:
        run_placements, pen_x, low_x = _shape_font_run(
            run, path, direction, script, language, features, variations,
            size, pen_x, low_x, run_index,
        )
        placements.extend(run_placements)
        run_index += 1
    return placements, pen_x, low_x, run_index


def _shape_font_run(
    run, path, direction, script, language, features, variations,
    size, pen_x, low_x, run_index,
):
    infos, positions = _shape_run(
        run, path, direction=direction, script=script, language=language,
        features=features, variations=variations,
    )
    scale = size / font_metrics(path)["units_per_em"]
    placements = []
    for info, position in zip(infos, positions):
        placements.append(
            (
                int(info.codepoint), path,
                pen_x + float(position.x_offset) * scale,
                float(position.y_offset) * scale,
                (run_index, int(info.cluster)),
            )
        )
        pen_x += float(position.x_advance) * scale
        low_x = min(low_x, pen_x)
    return placements, pen_x, low_x


def _finish_shaped_line(placements, pen_x, low_x, value, word_spacing, letter_spacing):
    if low_x:
        placements = [
            (gid, path, x - low_x, y, cluster)
            for gid, path, x, y, cluster in placements
        ]
    width = pen_x - low_x
    if word_spacing or letter_spacing:
        placements, spacing = _apply_text_spacing(
            placements, word_spacing=word_spacing, letter_spacing=letter_spacing
        )
        width += spacing
    return placements, max(width, 0.0), value


def _apply_text_spacing(placements, *, word_spacing, letter_spacing):
    offset = 0.0
    result = []
    for index, (glyph_id, path, x, y, cluster) in enumerate(placements):
        result.append((glyph_id, path, x + offset, y, cluster))
        if glyph_id == glyph_id_for_char(path, " "):
            offset += word_spacing
        if index + 1 < len(placements) and cluster != placements[index + 1][4]:
            offset += letter_spacing
    return result, offset


def _shape_with_options(value, paths, size, missing_glyph, options):
    return _shape_line(
        value,
        paths,
        size,
        missing_glyph,
        direction=options.get("direction"),
        script=options.get("script"),
        language=options.get("language"),
        features=options.get("features"),
        variations=options.get("variations"),
        word_spacing=options.get("word_spacing", 0.0),
        letter_spacing=options.get("letter_spacing", 0.0),
    )


def measure_glyph(char, *, size=64, path=None, fallbacks=(), variations=None):
    """Return a grapheme's shaped advance and glyph outline bounds in pixels."""
    if not isinstance(char, str) or not char:
        raise ValueError("char must be a non-empty string")
    if (
        isinstance(size, bool)
        or not isinstance(size, Real)
        or not isfinite(size)
        or size <= 0
    ):
        raise ValueError("size must be a finite positive number")
    paths = font_paths(path, fallbacks)
    shaped = _shape_with_options(
        char,
        paths,
        float(size),
        "ignore",
        {
            "variations": variations,
            "features": None,
            "direction": None,
            "script": None,
            "language": None,
        },
    )
    bounds = []
    variation_items = tuple(sorted((variations or {}).items()))
    for glyph_id, glyph_path, x, y, _cluster in shaped[0]:
        glyph = glyph_metrics(glyph_path, glyph_id, variation_items)
        scale = float(size) / font_metrics(glyph_path)["units_per_em"]
        x0, y0, x1, y1 = glyph["bbox"]
        bounds.append((x + x0 * scale, y + y0 * scale, x + x1 * scale, y + y1 * scale))
    ink_bounds = (
        (
            min(item[0] for item in bounds),
            min(item[1] for item in bounds),
            max(item[2] for item in bounds),
            max(item[3] for item in bounds),
        )
        if bounds
        else (0.0, 0.0, 0.0, 0.0)
    )
    return {"advance": shaped[1], "ink_bounds": ink_bounds}


def measure_text(
    value,
    *,
    size=64,
    path=None,
    fallbacks=(),
    box_width=None,
    box_height=None,
    line_height=1.2,
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
    overflow="visible",
    vertical_metrics="hhea",
):
    """Measure shaped text, returning its lines, line widths, width, and height."""
    _validate_measure_dimensions(size, line_height, box_width, box_height)
    paths = font_paths(path, fallbacks)
    lines = layout_text(
        value,
        paths,
        float(size),
        box_width=box_width,
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
        overflow=overflow,
    )
    return _measure_result(
        lines,
        paths,
        float(size),
        float(line_height),
        float(first_line_indent),
        box_width,
        box_height,
        variations,
        vertical_metrics,
    )


def _validate_measure_dimensions(size, line_height, box_width, box_height):
    _require_positive_finite(size, "text size")
    _require_positive_finite(line_height, "line_height")
    _require_optional_positive_finite(box_width, "box_width")
    _require_optional_positive_finite(box_height, "box_height")


def _require_positive_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def _require_optional_positive_finite(value, name):
    if value is not None:
        _require_positive_finite(value, name)


def _measure_result(
    lines,
    paths,
    size,
    line_height,
    first_line_indent,
    box_width,
    box_height,
    variations,
    vertical_metrics,
):
    ascent, descent = text_vertical_metrics(
        layout_font_paths(paths, lines), size, vertical_metrics
    )
    line_advance = float(size) * float(line_height)
    height = ascent - descent + max(len(lines) - 1, 0) * line_advance
    width = max(
        (line[1] + (float(first_line_indent) if line[5] else 0.0) for line in lines),
        default=0.0,
    )
    ink_bounds = _layout_ink_bounds(lines, size, line_height, variations)
    truncated = any(line[4] for line in lines)
    return {
        "width": width,
        "height": height,
        "overflow": (box_width is not None and width > box_width)
        or (box_height is not None and height > box_height)
        or truncated,
        "lines": tuple(line[2] for line in lines),
        "line_widths": tuple(line[1] for line in lines),
        "line_count": len(lines),
        "line_metrics": tuple(
            {
                "text": line[2],
                "width": line[1],
                "baseline": ascent - index * line_advance,
            }
            for index, line in enumerate(lines)
        ),
        "ascent": ascent,
        "descent": descent,
        "baseline": ascent,
        "ink_bounds": ink_bounds,
        "truncated": truncated,
    }


def _layout_ink_bounds(lines, size, line_height=1.2, variations=None):
    """Return glyph outline bounds as (left, bottom, right, top) pixels."""
    variation_items = tuple(sorted((variations or {}).items()))
    line_advance = size * line_height
    bounds = []
    for line_index, line in enumerate(lines):
        for glyph_id, path, x, y, _cluster in line[0]:
            metrics = glyph_metrics(path, glyph_id, variation_items)
            units = font_metrics(path)["units_per_em"]
            x0, y0, x1, y1 = metrics["bbox"]
            scale = size / units
            bounds.append(
                (
                    x + x0 * scale,
                    y + y0 * scale - line_index * line_advance,
                    x + x1 * scale,
                    y + y1 * scale - line_index * line_advance,
                )
            )
    if not bounds:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )
