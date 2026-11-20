"""Font file lookup, shaping backends, and OpenType glyph metrics."""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

import uharfbuzz as hb
from fontTools.ttLib import TTFont

_FONT_SUFFIXES = {".ttf", ".otf", ".ttc"}
_FAMILY_NAME_IDS = {1, 16}
_STYLE_NAME_IDS = {2, 17}
_UNDERLINE_OFFSET_FALLBACK = -3 / 40
_UNDERLINE_THICKNESS_FALLBACK = 1 / 20
_STRIKE_OFFSET_FALLBACK = 3 / 10
_STRIKE_THICKNESS_FALLBACK = 1 / 20
_UNITS_PER_EM = "units_per_em"


def default_font_path() -> str:
    """Return the packaged Roboto Regular font for text and glyph rendering."""
    return str(Path(__file__).with_name("assets") / "Roboto-Regular.ttf")


@lru_cache(maxsize=64)
def system_font(family="sans-serif", style="Regular"):
    """Resolve a system-installed font family to a font file path."""
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
    wanted = family.casefold(), style.casefold()
    for root in roots:
        if root.is_dir():
            match = next((path for path in root.rglob("*") if _matches_font(path, wanted)), None)
            if match is not None:
                return str(match)
    return None


def _matches_font(path, wanted):
    if path.suffix.casefold() not in _FONT_SUFFIXES:
        return False
    try:
        font = TTFont(path, lazy=True)
        try:
            family_names, style_names = _font_names(font)
        finally:
            font.close()
    except Exception:
        return False
    family, style = wanted
    return family in family_names and (style in style_names or style == "regular")


def _font_names(font):
    family_names, style_names = set(), set()
    for name in font["name"].names:
        if name.nameID in _FAMILY_NAME_IDS:
            family_names.add(name.toUnicode().casefold())
        elif name.nameID in _STYLE_NAME_IDS:
            style_names.add(name.toUnicode().casefold())
    return family_names, style_names


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
    face = hb.Face(Path(path).read_bytes())
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
    return {_UNITS_PER_EM: units, "ascent": ascent, "descent": descent}


def text_decoration_metrics(path, size):
    """Return standard OpenType underline and strike-through geometry in pixels."""
    font = _ttfont(path)
    units = float(font["head"].unitsPerEm)
    post = font.get("post")
    os2 = font.get("OS/2")
    scale = float(size) / units
    return {
        "underline_offset": float(getattr(post, "underlinePosition", units * _UNDERLINE_OFFSET_FALLBACK)) * scale,
        "underline_thickness": float(getattr(post, "underlineThickness", units * _UNDERLINE_THICKNESS_FALLBACK)) * scale,
        "strike_offset": float(getattr(os2, "yStrikeoutPosition", units * _STRIKE_OFFSET_FALLBACK)) * scale,
        "strike_thickness": float(getattr(os2, "yStrikeoutSize", units * _STRIKE_THICKNESS_FALLBACK)) * scale,
    }


def text_vertical_metrics(paths, size, vertical_metrics="hhea"):
    """Return the maximum ascent and minimum descent in rendered pixels."""
    metrics = [font_metrics(path, vertical_metrics) for path in paths]
    ascent = max(item["ascent"] * size / item[_UNITS_PER_EM] for item in metrics)
    descent = min(item["descent"] * size / item[_UNITS_PER_EM] for item in metrics)
    return ascent, descent


def layout_font_paths(paths, lines):
    used = {path for placements, _, _, _, _, _ in lines for _, path, _, _, _ in placements}
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
        bbox = _glyf_bounds(glyph)
    else:
        bbox = _variation_bounds(font, glyph_name, variation_items)
    return {"name": glyph_name, "advance": float(advance), "bbox": bbox}


def _glyf_bounds(glyph):
    if glyph.numberOfContours == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(float(getattr(glyph, edge)) for edge in ("xMin", "yMin", "xMax", "yMax"))


def _variation_bounds(font, glyph_name, variation_items):
    from fontTools.pens.boundsPen import BoundsPen

    glyph_set = font.getGlyphSet(location=dict(variation_items) or None)
    pen = BoundsPen(glyph_set)
    glyph_set[glyph_name].draw(pen)
    return tuple(map(float, pen.bounds or (0.0, 0.0, 0.0, 0.0)))


@lru_cache(maxsize=1024)
def glyph_id_for_char(path, char):
    glyph_name = _cmap(path).get(ord(char))
    return None if glyph_name is None else _ttfont(path).getGlyphID(glyph_name)
