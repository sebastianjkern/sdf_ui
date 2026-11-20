__docformat__ = "google"

from functools import lru_cache
from math import ceil
from numbers import Real

import numpy as np
from PIL import ImageFont

from sdf_ui.fonts import (
    font_metrics as _font_metrics,
)
from sdf_ui.fonts import (
    font_paths,
    layout_font_paths,
    layout_text,
    text_decoration_metrics,
    text_vertical_metrics,
)
from sdf_ui.fonts import (
    glyph_metrics as _glyph_metrics,
)
from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.primitives._text_composite import (
    _composite_glyph,
    _composite_rule,
)
from sdf_ui.gpu.nodes.primitives.glyph.plugin import (
    _contour_distance_sq,
    _contour_winding,
    _raw_glyph_contours_by_id,
)

_VARIATIONS_KEY = "variations"
_UNITS_PER_EM_KEY = "units_per_em"
_FIELD_KEY = "field"
_UNITS_PER_PIXEL_KEY = "units_per_pixel"
_ORIGIN_KEY = "origin"
_OVERLINE_POSITION = 0.9
_HINTED_DISTANCE_SCALE = 0.25
_GLYPH_PADDING_SCALE = 0.08
_MINIMUM_RENDER_EPSILON = 1e-6
_CFG_FONT_SIZE = "font_size"
_CFG_SAMPLES = "samples"
_CFG_CACHE_SIZE = "cache_size"
_CFG_LINE_HEIGHT = "line_height"
_CFG_OVERSAMPLE = "oversample"
_CFG_MIN_RENDER_SIZE = "min_render_size"
_CFG_ALIGN = "align"
_CFG_VALIGN = "valign"
_CFG_OVERFLOW = "overflow"
_CFG_TRIM_TRAILING_SPACES = "trim_trailing_spaces"
_CFG_WHITE_SPACE = "white_space"
_CFG_FIRST_LINE_INDENT = "first_line_indent"
_CFG_DECORATION_THICKNESS = "decoration_thickness"
_CFG_VERTICAL_METRICS = "vertical_metrics"
_CFG_MISSING_GLYPH = "missing_glyph"


def render_text(renderer, _inputs, params):
    ctx = renderer.ctx
    config = _TextRenderPipeline._text_render_config(ctx, params)
    layout = _TextRenderPipeline._text_layout(config)
    _TextRenderPipeline._draw_text_lines(config, layout)
    from sdf_ui.core.sdf import SDFTexture
    texture = ctx.r32f()
    texture.write(layout["field"].tobytes())
    return SDFTexture(tex=texture, context=ctx)


class _TextRenderPipeline:
    @staticmethod
    def _text_render_config(ctx, params):
        config = {
            "ctx": ctx,
            "params": params,
            "width": ctx.size[0],
            "height": ctx.size[1],
            _CFG_FONT_SIZE: float(params["size"]),
            "ox": float(params["ox"]),
            "oy": float(params["oy"]),
            "paths": font_paths(params["path"], params["fallback_fonts"]),
            _CFG_SAMPLES: max(1, round(int(params[_CFG_SAMPLES]) * getattr(ctx, "device_scale", 1.0))),
            _CFG_CACHE_SIZE: int(params[_CFG_CACHE_SIZE]),
            _CFG_LINE_HEIGHT: float(params[_CFG_LINE_HEIGHT]),
            _CFG_OVERSAMPLE: float(params[_CFG_OVERSAMPLE]),
            _CFG_MIN_RENDER_SIZE: float(params[_CFG_MIN_RENDER_SIZE]),
            "box": _normalize_box(params["box"]),
            _CFG_ALIGN: params[_CFG_ALIGN],
            _CFG_VALIGN: params[_CFG_VALIGN],
            _CFG_OVERFLOW: params[_CFG_OVERFLOW],
            "clip": params["clip"],
            _CFG_TRIM_TRAILING_SPACES: params[_CFG_TRIM_TRAILING_SPACES],
            _CFG_WHITE_SPACE: params.get(_CFG_WHITE_SPACE, "normal"),
            _CFG_FIRST_LINE_INDENT: float(params.get(_CFG_FIRST_LINE_INDENT, 0.0)),
            _CFG_DECORATION_THICKNESS: float(params.get(_CFG_DECORATION_THICKNESS, 0.0)),
            _CFG_VERTICAL_METRICS: params.get(_CFG_VERTICAL_METRICS, "hhea"),
            _CFG_MISSING_GLYPH: params[_CFG_MISSING_GLYPH],
        }
        decorations = params.get("decoration", ())
        if isinstance(decorations, str):
            decorations = () if decorations in {"none", ""} else (decorations,)
        _TextRenderPipeline._validate_decorations(decorations)
        config["decorations"] = decorations
        config["shape_options"] = _TextRenderPipeline._text_shape_options(params)
        _TextRenderPipeline._validate_text_options(config)
        return config


    @staticmethod
    def _text_shape_options(params):
        options = {
            key: params[key]
            for key in ("direction", "script", "language", "features", _VARIATIONS_KEY)
        }
        options["word_spacing"] = params.get("word_spacing", 0.0)
        options["letter_spacing"] = params.get("letter_spacing", 0.0)
        for key in ("features", _VARIATIONS_KEY):
            if isinstance(options[key], tuple):
                options[key] = dict(options[key])
        return options


    @staticmethod
    def _validate_decorations(decorations):
        if not isinstance(decorations, (tuple, list)) or any(
            item not in {"underline", "overline", "line-through"} for item in decorations
        ):
            raise ValueError("text decoration must be underline, overline, line-through, or a sequence")


    @staticmethod
    def _validate_text_options(config):
        _TextRenderPipeline._validate_text_dimensions(config)
        if not isinstance(config["clip"], bool):
            raise TypeError("text clip must be a bool")
        if not isinstance(config[_CFG_TRIM_TRAILING_SPACES], bool):
            raise TypeError("trim_trailing_spaces must be a bool")
        overflow = config[_CFG_OVERFLOW]
        if overflow not in {"visible", "clip", "error", "ellipsis"}:
            raise ValueError("unsupported text overflow mode")
        if config["clip"] and overflow not in {"visible", "clip"}:
            raise ValueError("clip=True conflicts with overflow='error'")
        if config["clip"]:
            config[_CFG_OVERFLOW] = "clip"
        if config[_CFG_OVERFLOW] != "visible" and config["box"] is None:
            raise ValueError("text overflow modes require a box")


    @staticmethod
    def _validate_text_dimensions(config):
        if not np.isfinite(config[_CFG_FONT_SIZE]) or config[_CFG_FONT_SIZE] <= 0:
            raise ValueError("text size must be a finite positive number")
        if not np.isfinite(config[_CFG_LINE_HEIGHT]) or config[_CFG_LINE_HEIGHT] <= 0:
            raise ValueError("text line_height must be a finite positive number")


    @staticmethod
    def _text_layout(config):
        width, height = config["width"], config["height"]
        font_size, box = config[_CFG_FONT_SIZE], config["box"]
        params = config["params"]
        lines = layout_text(
            params["value"], config["paths"], font_size,
            box_width=box[2] if box is not None else None,
            missing_glyph=config[_CFG_MISSING_GLYPH],
            trim_trailing_spaces=config[_CFG_TRIM_TRAILING_SPACES],
            white_space=config[_CFG_WHITE_SPACE],
            first_line_indent=config[_CFG_FIRST_LINE_INDENT],
            max_lines=params.get("max_lines"),
            overflow=config[_CFG_OVERFLOW],
            **config["shape_options"],
        )
        ascent, descent = text_vertical_metrics(
            layout_font_paths(config["paths"], lines), font_size, config[_CFG_VERTICAL_METRICS]
        )
        _TextRenderPipeline._validate_text_alignment(config)
        line_advance = font_size * config[_CFG_LINE_HEIGHT]
        first_baseline = _TextRenderPipeline._text_first_baseline(config, lines, ascent, descent, line_advance)
        return {
            "field": np.full((height, width), max(width, height), dtype=np.float32),
            "lines": lines,
            "ascent": ascent,
            "descent": descent,
            "line_advance": line_advance,
            "first_baseline": first_baseline,
            "glyph_cache_size": _glyph_cache_size(
                font_size, config[_CFG_OVERSAMPLE], config[_CFG_CACHE_SIZE], config[_CFG_MIN_RENDER_SIZE]
            ),
        }


    @staticmethod
    def _validate_text_alignment(config):
        if config[_CFG_ALIGN] not in {"left", "center", "right", "justify"}:
            raise ValueError("text align must be 'left', 'center', 'right', or 'justify'")
        if config[_CFG_VALIGN] not in {"top", "middle", "bottom"}:
            raise ValueError("text valign must be 'top', 'middle', or 'bottom'")
        if config["box"] is None and config[_CFG_ALIGN] != "left":
            raise ValueError("text alignment requires a box")


    @staticmethod
    def _text_first_baseline(config, lines, ascent, descent, line_advance):
        box = config["box"]
        if box is None:
            return config["oy"]
        x, y, box_width, box_height = box
        block_height = ascent - descent + max(len(lines) - 1, 0) * line_advance
        exceeds = block_height > box_height or any(
            line[1] + (config[_CFG_FIRST_LINE_INDENT] if line[5] else 0.0) > box_width
            for line in lines
        )
        if config[_CFG_OVERFLOW] == "error" and exceeds:
            raise ValueError("text layout exceeds the configured box")
        return _TextRenderPipeline._baseline_for_alignment(config[_CFG_VALIGN], y, box_height, ascent, descent, block_height, len(lines), line_advance)


    @staticmethod
    def _baseline_for_alignment(valign, y, box_height, ascent, descent, block_height, line_count, line_advance):
        remaining_lines = max(line_count - 1, 0) * line_advance
        if valign == "top":
            return y + box_height - ascent
        if valign == "middle":
            return y + (box_height - block_height) / 2 - descent + remaining_lines
        return y - descent + remaining_lines


    @staticmethod
    def _draw_text_lines(config, layout):
        for index, line in enumerate(layout["lines"]):
            line_data = _TextRenderPipeline._position_text_line(config, layout, line, index)
            _TextRenderPipeline._draw_line_glyphs(config, layout, line_data)
            _TextRenderPipeline._draw_line_decorations(config, layout, line_data)


    @staticmethod
    def _position_text_line(config, layout, line, index):
        placements, line_width, text, paragraph_last, _truncated, paragraph_first = line
        box = config["box"]
        x, y, box_width, box_height = box if box is not None else (
            config["ox"], config["oy"], None, None
        )
        if config[_CFG_ALIGN] == "justify":
            placements, line_width = _justify_placements(
                placements, text, line_width,
                box_width - (config[_CFG_FIRST_LINE_INDENT] if paragraph_first else 0.0),
                paragraph_last=paragraph_last,
            )
        line_x = _TextRenderPipeline._text_line_x(config, x, box_width, line_width, paragraph_first)
        baseline = layout["first_baseline"] - index * layout["line_advance"]
        clip_rect = (x, y, x + box_width, y + box_height) if config[_CFG_OVERFLOW] == "clip" else None
        return placements, line_width, line_x, baseline, clip_rect


    @staticmethod
    def _text_line_x(config, x, box_width, line_width, paragraph_first):
        if config["box"] is None:
            return config["ox"]
        if config[_CFG_ALIGN] in {"left", "justify"}:
            indent = config["params"].get(_CFG_FIRST_LINE_INDENT, 0.0) if paragraph_first else 0.0
            return x + indent
        if config[_CFG_ALIGN] == "center":
            return x + (box_width - line_width) / 2
        return x + box_width - line_width


    @staticmethod
    def _draw_line_glyphs(config, layout, line_data):
        placements, _, line_x, baseline, clip_rect = line_data
        variations = tuple(sorted((config["shape_options"][_VARIATIONS_KEY] or {}).items()))
        for glyph in placements:
            _TextRenderPipeline._draw_glyph(config, layout, glyph, line_x, baseline, clip_rect, variations)


    @staticmethod
    def _draw_glyph(config, layout, glyph, line_x, baseline, clip_rect, variations):
        glyph_id, path, glyph_x, glyph_y, _cluster = glyph
        metrics = _font_metrics(path)
        font_size = config[_CFG_FONT_SIZE]
        patch = None
        if font_size <= 64 and not variations:
            patch = _cached_hinted_glyph_sdf(path, glyph_id, max(1, round(font_size)), 4)
        if patch is None:
            glyph_scale = font_size / metrics[_UNITS_PER_EM_KEY]
            patch = _cached_glyph_sdf(path, glyph_id, config[_CFG_SAMPLES], layout["glyph_cache_size"], variations)
        else:
            glyph_scale = patch["hinted_size"] / metrics[_UNITS_PER_EM_KEY]
        if patch[_FIELD_KEY].size:
            _composite_glyph(
                layout["field"], patch, line_x + glyph_x, baseline + glyph_y,
                glyph_scale, glyph_scale, clip_rect=clip_rect,
            )


    @staticmethod
    def _draw_line_decorations(config, layout, line_data):
        _, line_width, line_x, baseline, clip_rect = line_data
        metrics = text_decoration_metrics(config["paths"][0], config[_CFG_FONT_SIZE])
        for decoration in config["decorations"]:
            offset, thickness = _TextRenderPipeline._decoration_metrics(
                decoration, metrics, layout["ascent"], config[_CFG_DECORATION_THICKNESS]
            )
            _composite_rule(layout["field"], line_x, baseline + offset, line_width, thickness, clip_rect)


    @staticmethod
    def _decoration_metrics(decoration, metrics, ascent, thickness):
        offset = {
            "underline": metrics["underline_offset"],
            "overline": ascent * _OVERLINE_POSITION,
            "line-through": metrics["strike_offset"],
        }[decoration]
        key = "strike_thickness" if decoration == "line-through" else "underline_thickness"
        return offset, thickness or metrics[key]

def _normalize_box(box):
    if box is None:
        return None
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        raise TypeError("text box must be (x, y, width, height)")
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
        for value in box
    ):
        raise ValueError("text box values must be finite numbers")
    x, y, width, height = map(float, box)
    if width <= 0 or height <= 0:
        raise ValueError("text box width and height must be positive")
    return x, y, width, height


def _glyph_cache_size(font_size, oversample, cache_size, min_render_size=0.0):
    effective_font_size = max(font_size, min_render_size)
    target_size = max(1, int(ceil(effective_font_size * max(oversample, 1.0))))
    if cache_size > 0:
        target_size = max(target_size, cache_size)
    return target_size


@lru_cache(maxsize=64)
def _space_glyph_id(path):
    from sdf_ui.fonts import glyph_id_for_char

    return glyph_id_for_char(path, " ")


def _justify_placements(
    placements, line_text, line_width, box_width, *, paragraph_last
):
    space_count = line_text.rstrip(" ").count(" ")
    if paragraph_last or space_count == 0 or box_width <= line_width:
        return placements, line_width

    extra_space = (box_width - line_width) / space_count
    spaces_before = 0
    adjusted = []
    for placement in placements:
        glyph_id, path, x, y = placement[:4]
        shifted = (glyph_id, path, x + spaces_before * extra_space, y)
        adjusted.append(shifted + placement[4:] if len(placement) > 4 else shifted)
        if glyph_id == _space_glyph_id(path) and spaces_before < space_count:
            spaces_before += 1
    return adjusted, line_width + spaces_before * extra_space


@lru_cache(maxsize=128)
def _glyph_character_map(path):
    """Return Unicode characters that map directly to each font glyph ID."""
    from fontTools.ttLib import TTFont

    font = TTFont(path, lazy=True)
    try:
        glyph_order = font.getGlyphOrder()
        name_to_id = {name: index for index, name in enumerate(glyph_order)}
        result = {}
        for codepoint, name in sorted(font.getBestCmap().items()):
            if codepoint < 32 or codepoint > 0x10FFFF or codepoint in {0x7F}:
                continue
            glyph_id = name_to_id.get(name)
            if glyph_id is not None:
                result.setdefault(glyph_id, chr(codepoint))
        return tuple(sorted(result.items()))
    finally:
        font.close()


@lru_cache(maxsize=1024)
def _cached_hinted_glyph_sdf(path, glyph_id, hinted_size, supersample):
    """Build a high-resolution SDF from FreeType-rendered glyph coverage."""
    character = dict(_glyph_character_map(path)).get(glyph_id)
    if character is None:
        return None

    scale = max(1, int(supersample))
    # Rasterize the outline at the higher working resolution. Upscaling a
    # bitmap rendered at `hinted_size` preserves its pixel grid, but not the
    # actual curve geometry; counters and joins in glyphs such as g and Q
    # become visibly lumpy as a result.
    raster_size = hinted_size * scale
    font = ImageFont.truetype(path, raster_size)
    mask, offset = font.getmask2(character, mode="L", anchor="ls")
    width, height = mask.size
    if width == 0 or height == 0:
        return _empty_hinted_glyph(path, raster_size, hinted_size)

    return _build_hinted_glyph_field(path, mask, offset, width, height, scale,
                                     raster_size, hinted_size)


def _empty_hinted_glyph(path, raster_size, hinted_size):
    return {
        _FIELD_KEY: np.empty((0, 0), dtype=np.float32),
        _ORIGIN_KEY: (0.0, 0.0),
        _UNITS_PER_PIXEL_KEY: _font_metrics(path)[_UNITS_PER_EM_KEY] / raster_size,
        "hinted_size": hinted_size,
    }


def _build_hinted_glyph_field(path, mask, offset, width, height, scale,
                              raster_size, hinted_size):

    coverage = np.asarray(mask, dtype=np.uint8).reshape(height, width)
    pad = 2 * scale
    alpha = np.pad(coverage, pad, mode="constant").astype(np.float32) / 255.0
    inside = (alpha >= 0.5).astype(np.uint8)
    field = _signed_mask_distance(inside)
    field = np.flipud(field).astype(np.float32, copy=True)
    units_per_pixel = _font_metrics(path)[_UNITS_PER_EM_KEY] / raster_size
    center = 0.5
    origin = ((float(offset[0]) - pad + center) * units_per_pixel,
              (-float(offset[1]) - height - pad + center) * units_per_pixel)
    return {_FIELD_KEY: field, _ORIGIN_KEY: origin,
            _UNITS_PER_PIXEL_KEY: units_per_pixel, "hinted_size": hinted_size}


def _signed_mask_distance(inside):
    import cv2
    inside_distance = cv2.distanceTransform(inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside_distance = cv2.distanceTransform(1 - inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return outside_distance - inside_distance + inside.astype(np.float32) - 0.5


@lru_cache(maxsize=1024)
def _cached_glyph_sdf(path, glyph_id, samples, glyph_cache_size, variation_items=()):
    raw_contours = _raw_glyph_contours_by_id(path, glyph_id, samples, variation_items)
    glyph_metrics = _glyph_metrics(path, glyph_id, variation_items)
    if not raw_contours or glyph_metrics["name"] is None:
        return _empty_glyph_sdf()

    font_metrics = _font_metrics(path)
    units_per_pixel = font_metrics[_UNITS_PER_EM_KEY] / glyph_cache_size
    pad = max(2, int(glyph_cache_size * _GLYPH_PADDING_SCALE))
    pad_units = pad * units_per_pixel
    x_min, y_min, x_max, y_max = glyph_metrics["bbox"]
    origin = (x_min - pad_units, y_min - pad_units)
    width = max(1, int(ceil((x_max - x_min) / units_per_pixel)) + pad * 2)
    height = max(1, int(ceil((y_max - y_min) / units_per_pixel)) + pad * 2)
    contours = _scale_glyph_contours(raw_contours, origin, units_per_pixel)
    inside = _rasterize_glyph_fill(contours, height, width)
    field = _signed_mask_distance(inside.astype(np.uint8))
    return {_FIELD_KEY: field, _ORIGIN_KEY: origin, _UNITS_PER_PIXEL_KEY: units_per_pixel}


def _scale_glyph_contours(raw_contours, origin, units_per_pixel):
    return tuple(
        tuple(((point[0] - origin[0]) / units_per_pixel,
               (point[1] - origin[1]) / units_per_pixel) for point in contour)
        for contour in raw_contours
    )


def _rasterize_glyph_fill(contours, height, width):
    yy, xx = np.mgrid[0:height, 0:width]
    points = np.stack((xx.astype(np.float32), yy.astype(np.float32)), axis=-1)
    distance_sq = np.full((height, width), np.inf, dtype=np.float32)
    winding = np.zeros((height, width), dtype=np.int16)

    for contour in contours:
        distance_sq = np.minimum(distance_sq, _contour_distance_sq(points, contour))
        winding += _contour_winding(points, contour)

    # Build distance from the filled silhouette rather than from every path
    # contour. Glyphs can contain overlapping contours; their internal edges
    # are not part of the filled shape boundary and must not create zero
    # distance seams through the rendered glyph.
    inside = winding != 0
    on_contour = distance_sq <= _MINIMUM_RENDER_EPSILON
    internal_contour = np.zeros_like(inside)
    internal_contour[1:-1, 1:-1] = (
        (inside[:-2, 1:-1] & inside[2:, 1:-1])
        | (inside[1:-1, :-2] & inside[1:-1, 2:])
    )
    inside |= on_contour & internal_contour
    return inside


def _empty_glyph_sdf():
    return {_FIELD_KEY: np.empty((0, 0), dtype=np.float32),
            _ORIGIN_KEY: (0.0, 0.0), _UNITS_PER_PIXEL_KEY: 1.0}


def register_plugins(registry):
    registry.register(
        Plugin(
            "text",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=(
                "value",
                "size",
                "ox",
                "oy",
                "path",
                _CFG_SAMPLES,
                _CFG_CACHE_SIZE,
                _CFG_LINE_HEIGHT,
                _CFG_OVERSAMPLE,
                _CFG_MIN_RENDER_SIZE,
                "box",
                _CFG_ALIGN,
                _CFG_VALIGN,
                "fallback_fonts",
                _CFG_MISSING_GLYPH,
                "clip",
                _CFG_OVERFLOW,
                _CFG_TRIM_TRAILING_SPACES,
                _CFG_WHITE_SPACE,
                _CFG_FIRST_LINE_INDENT,
                "max_lines",
                "decoration",
                _CFG_DECORATION_THICKNESS,
                "word_spacing",
                "letter_spacing",
                _CFG_VERTICAL_METRICS,
                "direction",
                "script",
                "language",
                "features",
                "variations",
            ),
            defaults={
                "size": 64,
                "ox": 0,
                "oy": 0,
                "path": None,
                _CFG_SAMPLES: 16,
                _CFG_CACHE_SIZE: 128,
                _CFG_LINE_HEIGHT: 1.2,
                _CFG_OVERSAMPLE: 2.5,
                _CFG_MIN_RENDER_SIZE: 64,
                "box": None,
                _CFG_ALIGN: "left",
                _CFG_VALIGN: "middle",
                "fallback_fonts": (),
                _CFG_MISSING_GLYPH: "ignore",
                "clip": False,
                _CFG_OVERFLOW: "visible",
                _CFG_TRIM_TRAILING_SPACES: True,
                _CFG_WHITE_SPACE: "normal",
                _CFG_FIRST_LINE_INDENT: 0.0,
                "max_lines": None,
                "decoration": (),
                _CFG_DECORATION_THICKNESS: 0.0,
                "word_spacing": 0.0,
                "letter_spacing": 0.0,
                _CFG_VERTICAL_METRICS: "hhea",
                "direction": None,
                "script": None,
                "language": None,
                "features": None,
                "variations": None,
            },
            public=True,
            render_func=render_text,
        )
    )
