"""Font discovery, OpenType shaping, and shared text layout."""

import os
import re
import subprocess
import unicodedata
from functools import lru_cache
from math import isfinite
from numbers import Real
from pathlib import Path

import uharfbuzz as hb
from bidi import algorithm as bidi_algorithm
from fontTools.ttLib import TTFont
from fontTools.unicodedata import script as unicode_script
from uniseg.linebreak import LB, line_break, line_break_units


def default_font_path() -> str:
    """Return the packaged Roboto Regular font for text and glyph rendering."""
    return str(Path(__file__).with_name("assets") / "Roboto-Regular.ttf")


@lru_cache(maxsize=64)
def system_font(family="sans-serif", style="Regular"):
    """Resolve a system-installed font family to a font file path.

    Uses Fontconfig where available (Linux and other Fontconfig platforms).
    The bundled font remains the deterministic default; callers opt in with
    ``path=system_font(...)`` or ``path="system:Family Name"``.
    """
    if not isinstance(family, str) or not family.strip():
        raise ValueError("font family must be a non-empty string")
    if not isinstance(style, str) or not style.strip():
        raise ValueError("font style must be a non-empty string")
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}", f"{family}:style={style}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as error:
        path = _scan_system_font(family, style)
        if path is None:
            raise RuntimeError(
                "system font lookup failed; install Fontconfig or pass a font file path"
            ) from error
        return path
    path = result.stdout.strip()
    if not path or not Path(path).is_file():
        raise LookupError(f"system font {family!r} ({style}) was not found")
    return path


def _scan_system_font(family, style):
    roots = [Path.home() / ".fonts", Path.home() / ".local/share/fonts"]
    if os.name == "nt":
        roots.extend(
            [
                Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
                Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts",
            ]
        )
    elif __import__("sys").platform == "darwin":
        roots.extend([Path("/System/Library/Fonts"), Path("/Library/Fonts")])
    else:
        roots.extend([Path("/usr/share/fonts"), Path("/usr/local/share/fonts")])
    wanted_family = family.casefold()
    wanted_style = style.casefold()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix.casefold() not in {".ttf", ".otf", ".ttc"}:
                continue
            try:
                font = TTFont(path, lazy=True)
                names = font["name"].names
                family_names = set()
                style_names = set()
                for name in names:
                    if name.nameID not in {1, 2, 16, 17}:
                        continue
                    text = name.toUnicode().casefold()
                    (family_names if name.nameID in {1, 16} else style_names).add(text)
                font.close()
                if wanted_family in family_names and (
                    wanted_style in style_names or wanted_style == "regular"
                ):
                    return str(path)
            except Exception:
                continue
    return None


def font_paths(path=None, fallbacks=()):
    """Normalize a primary font and ordered fallback font paths."""
    primary = str(path or default_font_path())
    if primary.startswith("system:"):
        primary = system_font(primary.partition(":")[2])
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


@lru_cache(maxsize=192)
def font_metrics(path, vertical_metrics="hhea"):
    font = _ttfont(path)
    units = float(font["head"].unitsPerEm)
    os2 = font.get("OS/2")
    if vertical_metrics == "hhea" or os2 is None:
        ascent = float(font["hhea"].ascent)
        descent = float(font["hhea"].descent)
    elif vertical_metrics == "typographic":
        ascent = float(getattr(os2, "sTypoAscender", font["hhea"].ascent))
        descent = float(getattr(os2, "sTypoDescender", font["hhea"].descent))
    elif vertical_metrics == "win":
        ascent = float(getattr(os2, "usWinAscent", font["hhea"].ascent))
        descent = -float(getattr(os2, "usWinDescent", -font["hhea"].descent))
    else:
        raise ValueError("vertical_metrics must be 'hhea', 'typographic', or 'win'")
    return {
        "units_per_em": units,
        "ascent": ascent,
        "descent": descent,
    }


def text_decoration_metrics(path, size):
    """Return standard OpenType underline and strike-through geometry in pixels."""
    font = _ttfont(path)
    units = float(font["head"].unitsPerEm)
    post = font.get("post")
    os2 = font.get("OS/2")
    scale = float(size) / units
    return {
        "underline_offset": float(getattr(post, "underlinePosition", -units * 0.075))
        * scale,
        "underline_thickness": float(getattr(post, "underlineThickness", units * 0.05))
        * scale,
        "strike_offset": float(getattr(os2, "yStrikeoutPosition", units * 0.3)) * scale,
        "strike_thickness": float(getattr(os2, "yStrikeoutSize", units * 0.05)) * scale,
    }


def text_vertical_metrics(paths, size, vertical_metrics="hhea"):
    """Return the maximum ascent and minimum descent in rendered pixels."""
    metrics = [font_metrics(path, vertical_metrics) for path in paths]
    ascent = max(item["ascent"] * size / item["units_per_em"] for item in metrics)
    descent = min(item["descent"] * size / item["units_per_em"] for item in metrics)
    return ascent, descent


def layout_font_paths(paths, lines):
    used = {
        path for placements, _, _, _, _, _ in lines for _, path, _, _, _ in placements
    }
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


@lru_cache(maxsize=1024)
def glyph_id_for_char(path, char):
    glyph_name = _cmap(path).get(ord(char))
    return None if glyph_name is None else _ttfont(path).getGlyphID(glyph_name)


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


def _isolate_tokens(value):
    tokens = []
    index = 0
    while index < len(value):
        char = value[index]
        if ord(char) in {0x2066, 0x2067, 0x2068}:
            depth = 1
            end = index + 1
            while end < len(value) and depth:
                code = ord(value[end])
                if code in {0x2066, 0x2067, 0x2068}:
                    depth += 1
                elif code == 0x2069:
                    depth -= 1
                end += 1
            if depth == 0:
                inner = value[index + 1 : end - 1]
                isolate_type = ord(char)
                base_level = (
                    1
                    if isolate_type == 0x2067
                    else 0
                    if isolate_type == 0x2066
                    else bidi_algorithm.get_base_level(inner)
                )
                tokens.append(("isolate", inner, base_level))
                index = end
                continue
            index += 1
            continue
        if ord(char) != 0x2069:
            tokens.append(("char", char, None))
        index += 1
    return tokens


def _bidi_runs(value, direction, base_level=None):
    """Return visual-order logical text runs with resolved bidi directions."""
    if direction in {"ltr", "rtl"}:
        return [(value, direction)]
    tokens = _isolate_tokens(value)
    bidi_text = "".join(
        token[1] if token[0] == "char" else "א" if token[2] else "A" for token in tokens
    )
    storage = bidi_algorithm.get_empty_storage()
    if base_level is None:
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

    def flush():
        if current:
            indexes = sorted(item["_source_index"] for item in current)
            run_text = "".join(tokens[index][1] for index in indexes)
            runs.append((run_text, "rtl" if level % 2 else "ltr"))
            current.clear()

    for item in storage["chars"]:
        item_level = item["level"]
        token = tokens[item["_source_index"]]
        if token[0] == "isolate":
            flush()
            runs.extend(_bidi_runs(token[1], None, base_level=token[2]))
            level = None
            continue
        if current and item_level != level:
            flush()
        current.append(item)
        level = item_level
    flush()
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
    word_spacing=0.0,
    letter_spacing=0.0,
):
    # Select a font per grapheme, then shape each maximal font run as a whole so
    # contextual substitutions and mark positioning can cross character boundaries.
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
                    placements.append(
                        (
                            glyph_id,
                            path,
                            pen_x + x_offset,
                            y_offset,
                            (run_index, int(info.cluster)),
                        )
                    )
                    pen_x += float(position.x_advance) * scale
                    low_x = min(low_x, pen_x)
                run_index += 1

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
    if not isinstance(value, str):
        raise TypeError("text value must be a string")
    if missing_glyph not in {"ignore", "error"}:
        raise ValueError("missing_glyph must be 'ignore' or 'error'")
    if direction not in {None, "ltr", "rtl"}:
        raise ValueError("text direction must be None, 'ltr', or 'rtl'")
    if white_space not in {"normal", "pre", "pre-wrap"}:
        raise ValueError("white_space must be 'normal', 'pre', or 'pre-wrap'")
    if trim_trailing_spaces is None:
        trim_trailing_spaces = white_space == "normal"
    elif not isinstance(trim_trailing_spaces, bool):
        raise TypeError("trim_trailing_spaces must be a bool or None")
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
    if max_lines is not None and (
        isinstance(max_lines, bool) or not isinstance(max_lines, int) or max_lines < 1
    ):
        raise ValueError("max_lines must be a positive integer")
    if overflow not in {"visible", "clip", "error", "ellipsis"}:
        raise ValueError("unsupported text overflow mode")
    if overflow == "ellipsis" and max_lines is None:
        raise ValueError("ellipsis overflow requires max_lines")
    options = {
        "direction": direction,
        "script": script,
        "language": language,
        "features": features,
        "variations": variations,
        "word_spacing": float(word_spacing),
        "letter_spacing": float(letter_spacing),
    }

    def shape(text):
        return _shape_with_options(text, paths, size, missing_glyph, options)

    lines = []
    for paragraph in re.split(r"\r\n|[\r\n\v\f\x85\u2028\u2029]", value):
        if white_space == "normal":
            if trim_trailing_spaces:
                paragraph = re.sub(r"[ \t]+", " ", paragraph).strip(" \t")
            else:
                trailing = re.search(r"[ \t]+$", paragraph)
                trailing_spaces = trailing.group() if trailing else ""
                paragraph = re.sub(r"[ \t]+", " ", paragraph.rstrip(" \t"))
                paragraph += trailing_spaces
        else:
            paragraph = _expand_tabs(paragraph)
        if not paragraph:
            lines.append((*shape(""), True, False, True))
            continue
        should_wrap = box_width is not None and white_space != "pre"
        if should_wrap:
            wrapped = _wrap_paragraph(
                paragraph,
                shape,
                box_width,
                trim_trailing_spaces=trim_trailing_spaces,
                first_line_indent=first_line_indent,
            )
        else:
            text = paragraph.rstrip(" \t") if trim_trailing_spaces else paragraph
            wrapped = [shape(text)]
        lines.extend(
            (*line, index == len(wrapped) - 1, False, index == 0)
            for index, line in enumerate(wrapped)
        )

    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        placements, width, text, _last, _truncated, first = lines[-1]
        if overflow == "ellipsis":
            marker = "…" if any(0x2026 in _cmap(path) for path in paths) else "..."
            clusters = list(_graphemes(text.rstrip(" \t")))
            available_width = (
                None
                if box_width is None
                else box_width - (first_line_indent if first else 0.0)
            )
            while True:
                candidate = "".join(clusters) + marker
                ellipsized = shape(candidate)
                if available_width is None or ellipsized[1] <= available_width:
                    break
                if not clusters:
                    break
                clusters.pop()
            lines[-1] = (*ellipsized, True, True, first)
        else:
            lines[-1] = (placements, width, text, True, True, first)
    return lines


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
        candidate = line + unit
        measured = candidate.rstrip() if trim_trailing_spaces else candidate
        if shape(measured)[1] <= line_width:
            line = candidate
            continue
        if line:
            text = line.rstrip() if trim_trailing_spaces else line
            if text:
                lines.append(shape(text))
            line = ""

        if trim_trailing_spaces:
            unit = unit.lstrip()
        if not unit:
            continue
        measured = unit.rstrip() if trim_trailing_spaces else unit
        line_width = box_width - first_line_indent if not lines else box_width
        if shape(measured)[1] <= line_width:
            line = unit
            continue
        chunks = _wrap_long_word(unit, shape, line_width, box_width)
        lines.extend(shape(chunk) for chunk in chunks[:-1])
        line = chunks[-1]

    if line or not lines:
        text = line.rstrip() if trim_trailing_spaces else line
        lines.append(shape(text))
    return lines


def _wrap_long_word(word, shape, box_width, continuation_width=None):
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
    if box_height is not None and (
        isinstance(box_height, bool)
        or not isinstance(box_height, Real)
        or not isfinite(box_height)
        or box_height <= 0
    ):
        raise ValueError("box_height must be a finite positive number")
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
    ascent, descent = text_vertical_metrics(
        layout_font_paths(paths, lines), float(size), vertical_metrics
    )
    line_advance = float(size) * float(line_height)
    height = ascent - descent + max(len(lines) - 1, 0) * line_advance
    width = max(
        (line[1] + (float(first_line_indent) if line[5] else 0.0) for line in lines),
        default=0.0,
    )
    ink_bounds = _layout_ink_bounds(lines, float(size), float(line_height), variations)
    return {
        "width": width,
        "height": height,
        "overflow": (box_width is not None and width > box_width)
        or (box_height is not None and height > box_height)
        or any(line[4] for line in lines),
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
        "truncated": any(line[4] for line in lines),
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
