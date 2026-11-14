"""Font discovery, OpenType shaping, and shared text layout."""

import unicodedata
from functools import lru_cache
from math import isfinite
from numbers import Real
from pathlib import Path

import uharfbuzz as hb
from bidi import algorithm as bidi_algorithm
from fontTools.ttLib import TTFont
from fontTools.unicodedata import script as unicode_script


def default_font_path() -> str:
    """Return the packaged Roboto Regular font for text and glyph rendering."""
    return str(Path(__file__).with_name("assets") / "Roboto-Regular.ttf")


def font_paths(path=None, fallbacks=()):
    """Normalize a primary font and ordered fallback font paths."""
    primary = str(path or default_font_path())
    if isinstance(fallbacks, (str, Path)):
        fallbacks = (fallbacks,)
    elif not isinstance(fallbacks, (tuple, list)):
        raise TypeError("fallback_fonts must be a font path or a sequence of paths")
    if any(not isinstance(item, (str, Path)) for item in fallbacks):
        raise TypeError("fallback font entries must be paths")
    return tuple(dict.fromkeys((primary, *(str(item) for item in fallbacks))))


@lru_cache(maxsize=64)
def _ttfont(path):
    return TTFont(path, lazy=False)


@lru_cache(maxsize=256)
def _hb_font(path, variation_items=()):
    data = Path(path).read_bytes()
    face = hb.Face(data)
    font = hb.Font(face)
    font.scale = (face.upem, face.upem)
    hb.ot_font_set_funcs(font)
    if variation_items:
        font.set_variations(dict(variation_items))
    return font


@lru_cache(maxsize=64)
def _cmap(path):
    font = _ttfont(path)
    result = {}
    for table in font["cmap"].tables:
        if table.isUnicode():
            result.update(table.cmap)
    return result


@lru_cache(maxsize=64)
def font_metrics(path):
    font = _ttfont(path)
    units = float(font["head"].unitsPerEm)
    return {
        "units_per_em": units,
        "ascent": float(font["hhea"].ascent),
        "descent": float(font["hhea"].descent),
    }


def text_vertical_metrics(paths, size):
    """Return the maximum ascent and minimum descent in rendered pixels."""
    metrics = [font_metrics(path) for path in paths]
    ascent = max(item["ascent"] * size / item["units_per_em"] for item in metrics)
    descent = min(item["descent"] * size / item["units_per_em"] for item in metrics)
    return ascent, descent


def layout_font_paths(paths, lines):
    used = {path for placements, _, _ in lines for _, path, _, _ in placements}
    selected = tuple(path for path in paths if path in used)
    return selected or (paths[0],)


@lru_cache(maxsize=4096)
def glyph_metrics(path, glyph_id, variation_items=()):
    """Get outline metrics for a shaped glyph ID."""
    font = _ttfont(path)
    glyph_order = font.getGlyphOrder()
    if glyph_id < 0 or glyph_id >= len(glyph_order):
        return {"name": None, "advance": 0.0, "bbox": (0.0, 0.0, 0.0, 0.0)}
    glyph_name = glyph_order[glyph_id]
    advance, _bearing = font["hmtx"].metrics.get(glyph_name, (0, 0))
    if "glyf" in font and not variation_items:
        glyph = font["glyf"][glyph_name]
        bbox = (
            (0.0, 0.0, 0.0, 0.0)
            if glyph.numberOfContours == 0
            else (
                float(glyph.xMin),
                float(glyph.yMin),
                float(glyph.xMax),
                float(glyph.yMax),
            )
        )
    else:
        from fontTools.pens.boundsPen import BoundsPen

        glyph_set = font.getGlyphSet(location=dict(variation_items) or None)
        pen = BoundsPen(glyph_set)
        glyph_set[glyph_name].draw(pen)
        bbox = tuple(map(float, pen.bounds or (0.0, 0.0, 0.0, 0.0)))
    return {"name": glyph_name, "advance": float(advance), "bbox": bbox}


@lru_cache(maxsize=2048)
def _graphemes(value):
    """Split text into fallback-safe grapheme-like clusters."""
    clusters = []
    current = ""
    regional_count = 0
    join_next = False
    for char in value:
        code = ord(char)
        category = unicodedata.category(char)
        is_variation = 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF
        is_regional = 0x1F1E6 <= code <= 0x1F1FF
        extender = category in {"Mn", "Mc", "Me"} or is_variation or code == 0x200D
        if (
            current
            and not extender
            and not join_next
            and not (is_regional and regional_count % 2 == 1)
        ):
            clusters.append(current)
            current = ""
            regional_count = 0
        current += char
        if is_regional:
            regional_count += 1
        elif not extender:
            regional_count = 0
        join_next = code == 0x200D
    if current:
        clusters.append(current)
    return tuple(clusters)


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


def _is_variation_selector(char):
    code = ord(char)
    return 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF


def _bidi_runs(value, direction):
    """Return visual-order logical text runs with resolved bidi directions."""
    if direction in {"ltr", "rtl"}:
        return [(value, direction)]
    source_indexes = [
        index
        for index, char in enumerate(value)
        if ord(char) not in {0x2066, 0x2067, 0x2068, 0x2069}
    ]
    bidi_text = "".join(value[index] for index in source_indexes)
    storage = bidi_algorithm.get_empty_storage()
    base_level = bidi_algorithm.get_base_level(bidi_text)
    storage["base_level"] = base_level
    storage["base_dir"] = ("L", "R")[base_level]
    bidi_algorithm.get_embedding_levels(bidi_text, storage)
    for index, char in enumerate(storage["chars"]):
        char["_source_index"] = index
    bidi_algorithm.explicit_embed_and_overrides(storage)
    bidi_algorithm.resolve_weak_types(storage)
    bidi_algorithm.resolve_neutral_types(storage, False)
    bidi_algorithm.resolve_implicit_levels(storage, False)
    bidi_algorithm.reorder_resolved_levels(storage, False)

    runs = []
    current = []
    level = None
    for item in storage["chars"]:
        item_level = item["level"]
        if current and item_level != level:
            indexes = sorted(item["_source_index"] for item in current)
            runs.append(
                (
                    "".join(value[source_indexes[index]] for index in indexes),
                    "rtl" if level % 2 else "ltr",
                )
            )
            current = []
        current.append(item)
        level = item_level
    if current:
        indexes = sorted(item["_source_index"] for item in current)
        runs.append(
            (
                "".join(value[source_indexes[index]] for index in indexes),
                "rtl" if level % 2 else "ltr",
            )
        )
    return runs or [(value, "ltr")]


def _script_runs(value, forced_script=None):
    if forced_script:
        return [(value, forced_script)]
    scripts = [unicode_script(char) for char in value]
    for index, script in enumerate(scripts):
        if script not in {"Zyyy", "Zinh", "Zzzz"}:
            continue
        neighbor = next(
            (
                item
                for item in scripts[index + 1 :]
                if item not in {"Zyyy", "Zinh", "Zzzz"}
            ),
            None,
        )
        previous = next(
            (
                item
                for item in reversed(scripts[:index])
                if item not in {"Zyyy", "Zinh", "Zzzz"}
            ),
            None,
        )
        scripts[index] = previous or neighbor or "Zyyy"
    runs = []
    for char, script in zip(value, scripts):
        if runs and runs[-1][0] == script:
            runs[-1] = (script, runs[-1][1] + char)
        else:
            runs.append((script, char))
    return [(text, script) for script, text in runs]


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
):
    # Select a font per grapheme, then shape each maximal font run as a whole so
    # contextual substitutions and mark positioning can cross character boundaries.
    visual_runs = _bidi_runs(value, direction)
    placements = []
    pen_x = 0.0
    low_x = 0.0
    for visual_run, run_direction in visual_runs:
        script_runs = _script_runs(visual_run, script)
        if run_direction == "rtl":
            script_runs.reverse()
        for script_run, run_script in script_runs:
            font_runs = []
            for cluster in _graphemes(script_run):
                path = _font_for_cluster(cluster, paths, missing_glyph)
                if path is None:
                    continue
                if font_runs and font_runs[-1][0] == path:
                    font_runs[-1] = (path, font_runs[-1][1] + cluster)
                else:
                    font_runs.append((path, cluster))
            if run_direction == "rtl":
                font_runs.reverse()
            for path, run in font_runs:
                glyph_infos, positions = _shape_run(
                    run,
                    path,
                    direction=run_direction,
                    script=run_script,
                    language=language,
                    features=features,
                    variations=variations,
                )
                scale = size / font_metrics(path)["units_per_em"]
                for info, position in zip(glyph_infos, positions):
                    glyph_id = int(info.codepoint)
                    x_offset = float(position.x_offset) * scale
                    y_offset = float(position.y_offset) * scale
                    placements.append((glyph_id, path, pen_x + x_offset, y_offset))
                    pen_x += float(position.x_advance) * scale
                    low_x = min(low_x, pen_x)

    if low_x:
        placements = [(gid, path, x - low_x, y) for gid, path, x, y in placements]
    return placements, pen_x - low_x, value


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
    )


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
):
    """Return wrapped lines as positioned OpenType glyph IDs and pixel advances."""
    if not isinstance(value, str):
        raise TypeError("text value must be a string")
    if missing_glyph not in {"ignore", "error"}:
        raise ValueError("missing_glyph must be 'ignore' or 'error'")
    if direction not in {None, "ltr", "rtl"}:
        raise ValueError("text direction must be None, 'ltr', or 'rtl'")
    options = {
        "direction": direction,
        "script": script,
        "language": language,
        "features": features,
        "variations": variations,
    }

    def shape(text):
        return _shape_with_options(text, paths, size, missing_glyph, options)

    lines = []
    for paragraph in value.split("\n"):
        if box_width is None:
            lines.append(shape(paragraph))
            continue
        words = paragraph.split()
        if not words:
            lines.append(shape(""))
            continue
        line = ""
        for word in words:
            candidate = f"{line} {word}" if line else word
            shaped = shape(candidate)
            if shaped[1] <= box_width:
                line = candidate
                continue
            if line:
                lines.append(shape(line))
                line = ""
            if shape(word)[1] <= box_width:
                line = word
                continue
            chunks = _wrap_long_word(word, shape, box_width)
            lines.extend(shape(chunk) for chunk in chunks[:-1])
            line = chunks[-1]
        lines.append(shape(line))
    return lines


def _wrap_long_word(word, shape, box_width):
    chunks = []
    chunk = ""
    for cluster in _graphemes(word):
        candidate = chunk + cluster
        if chunk and shape(candidate)[1] > box_width:
            chunks.append(chunk)
            chunk = cluster
        else:
            chunk = candidate
    if chunk:
        chunks.append(chunk)
    return chunks


def measure_text(
    value,
    *,
    size=64,
    path=None,
    fallbacks=(),
    box_width=None,
    line_height=1.2,
    missing_glyph="ignore",
    direction=None,
    script=None,
    language=None,
    features=None,
    variations=None,
):
    """Measure shaped text, returning its lines, line widths, width, and height."""
    if (
        isinstance(size, bool)
        or not isinstance(size, Real)
        or not isfinite(size)
        or size <= 0
    ):
        raise ValueError("text size must be a finite positive number")
    if (
        isinstance(line_height, bool)
        or not isinstance(line_height, Real)
        or not isfinite(line_height)
        or line_height <= 0
    ):
        raise ValueError("line_height must be a finite positive number")
    if box_width is not None and (
        isinstance(box_width, bool)
        or not isinstance(box_width, Real)
        or not isfinite(box_width)
        or box_width <= 0
    ):
        raise ValueError("box_width must be a finite positive number")
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
    )
    ascent, descent = text_vertical_metrics(
        layout_font_paths(paths, lines), float(size)
    )
    line_advance = float(size) * float(line_height)
    height = ascent - descent + max(len(lines) - 1, 0) * line_advance
    return {
        "width": max((width for _, width, _ in lines), default=0.0),
        "height": height,
        "lines": tuple(line for _, _, line in lines),
        "line_widths": tuple(width for _, width, _ in lines),
    }
