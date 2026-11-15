__docformat__ = "google"

from functools import lru_cache
from math import ceil, floor
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
from sdf_ui.gpu.nodes.primitives.glyph.plugin import (
    _contour_distance_sq,
    _contour_winding,
    _raw_glyph_contours_by_id,
)


def render_text(renderer, _inputs, params):
    ctx = renderer.ctx
    width, height = ctx.size
    font_size = float(params["size"])
    ox = float(params["ox"])
    oy = float(params["oy"])
    paths = font_paths(params["path"], params["fallback_fonts"])
    samples = max(1, round(int(params["samples"]) * getattr(ctx, "device_scale", 1.0)))
    cache_size = int(params["cache_size"])
    line_height = float(params["line_height"])
    oversample = float(params["oversample"])
    min_render_size = float(params["min_render_size"])
    box = _normalize_box(params["box"])
    align = params["align"]
    valign = params["valign"]
    missing_glyph = params["missing_glyph"]
    overflow = params["overflow"]
    clip_alias = params["clip"]
    trim_trailing_spaces = params["trim_trailing_spaces"]
    white_space = params.get("white_space", "normal")
    first_line_indent = float(params.get("first_line_indent", 0.0))
    decorations = params.get("decoration", ())
    if isinstance(decorations, str):
        decorations = () if decorations in {"none", ""} else (decorations,)
    if not isinstance(decorations, (tuple, list)) or any(
        item not in {"underline", "overline", "line-through"} for item in decorations
    ):
        raise ValueError(
            "text decoration must be underline, overline, line-through, or a sequence"
        )
    decoration_thickness = float(params.get("decoration_thickness", 0.0))
    vertical_metrics_mode = params.get("vertical_metrics", "hhea")
    shape_options = {
        key: params[key]
        for key in ("direction", "script", "language", "features", "variations")
    }
    shape_options["word_spacing"] = params.get("word_spacing", 0.0)
    shape_options["letter_spacing"] = params.get("letter_spacing", 0.0)
    for key in ("features", "variations"):
        if isinstance(shape_options[key], tuple):
            shape_options[key] = dict(shape_options[key])

    if not np.isfinite(font_size) or font_size <= 0:
        raise ValueError("text size must be a finite positive number")
    if not np.isfinite(line_height) or line_height <= 0:
        raise ValueError("text line_height must be a finite positive number")
    if not isinstance(clip_alias, bool):
        raise TypeError("text clip must be a bool")
    if not isinstance(trim_trailing_spaces, bool):
        raise TypeError("trim_trailing_spaces must be a bool")
    if overflow not in {"visible", "clip", "error", "ellipsis"}:
        raise ValueError("unsupported text overflow mode")
    if clip_alias and overflow not in {"visible", "clip"}:
        raise ValueError("clip=True conflicts with overflow='error'")
    if clip_alias:
        overflow = "clip"
    if overflow != "visible" and box is None:
        raise ValueError("text overflow modes require a box")

    effective_font_size = max(font_size, min_render_size)
    field = np.full((height, width), max(width, height), dtype=np.float32)
    glyph_cache_size = _glyph_cache_size(
        font_size, oversample, cache_size, min_render_size
    )

    line_advance = font_size * line_height
    lines = layout_text(
        params["value"],
        paths,
        font_size,
        box_width=box[2] if box is not None else None,
        missing_glyph=missing_glyph,
        trim_trailing_spaces=trim_trailing_spaces,
        white_space=white_space,
        first_line_indent=first_line_indent,
        max_lines=params.get("max_lines"),
        overflow=overflow,
        **shape_options,
    )
    ascent, descent = text_vertical_metrics(
        layout_font_paths(paths, lines), font_size, vertical_metrics_mode
    )

    if align not in {"left", "center", "right", "justify"}:
        raise ValueError("text align must be 'left', 'center', 'right', or 'justify'")
    if valign not in {"top", "middle", "bottom"}:
        raise ValueError("text valign must be 'top', 'middle', or 'bottom'")
    if box is None and align != "left":
        raise ValueError("text alignment requires a box")

    x, y, box_width, box_height = box if box is not None else (ox, oy, None, None)
    if box is None:
        first_baseline = oy
    else:
        block_height = ascent - descent + max(len(lines) - 1, 0) * line_advance
        exceeds_box = block_height > box_height or any(
            line[1] + (first_line_indent if line[5] else 0.0) > box_width
            for line in lines
        )
        if overflow == "error" and exceeds_box:
            raise ValueError("text layout exceeds the configured box")
        if valign == "top":
            first_baseline = y + box_height - ascent
        elif valign == "middle":
            block_bottom = y + (box_height - block_height) / 2
            first_baseline = (
                block_bottom - descent + max(len(lines) - 1, 0) * line_advance
            )
        else:
            first_baseline = y - descent + max(len(lines) - 1, 0) * line_advance

    for line_index, (
        placements,
        line_width,
        line_text,
        paragraph_last,
        _truncated,
        paragraph_first,
    ) in enumerate(lines):
        if align == "justify":
            placements, line_width = _justify_placements(
                placements,
                line_text,
                line_width,
                box_width - (first_line_indent if paragraph_first else 0.0),
                paragraph_last=paragraph_last,
            )
        if box is None:
            line_x = ox
        elif align in {"left", "justify"}:
            line_x = x + (
                params.get("first_line_indent", 0.0) if paragraph_first else 0.0
            )
        elif align == "center":
            line_x = x + (box_width - line_width) / 2
        else:
            line_x = x + box_width - line_width

        baseline = first_baseline - line_index * line_advance
        for glyph_id, glyph_path, glyph_x, glyph_y, _cluster in placements:
            glyph_metrics = _font_metrics(glyph_path)
            variations = tuple(sorted((shape_options["variations"] or {}).items()))
            patch = None
            if font_size <= 64 and not variations:
                patch = _cached_hinted_glyph_sdf(
                    glyph_path, glyph_id, max(1, round(font_size)), 4
                )
            if patch is None:
                glyph_scale = font_size / glyph_metrics["units_per_em"]
                # Distance values must follow the same transform as glyph
                # geometry. Using effective_font_size here widened the
                # fill shader's antialias ramp whenever size < min_render_size.
                glyph_distance_scale = glyph_scale
                patch = _cached_glyph_sdf(
                    glyph_path,
                    glyph_id,
                    samples,
                    glyph_cache_size,
                    variations,
                )
            else:
                # Hinting is performed at the target pixel size. Keep the
                # hinted shape at that size instead of scaling it to the
                # large outline-SDF cache's minimum render size.
                glyph_scale = patch["hinted_size"] / glyph_metrics["units_per_em"]
                glyph_distance_scale = glyph_scale
            if patch["field"].size:
                _composite_glyph(
                    field,
                    patch,
                    line_x + glyph_x,
                    baseline + glyph_y,
                    glyph_scale,
                    glyph_distance_scale,
                    clip_rect=(x, y, x + box_width, y + box_height)
                    if overflow == "clip"
                    else None,
                )

        decoration_metrics = text_decoration_metrics(paths[0], font_size)
        for decoration in decorations:
            offset = {
                "underline": decoration_metrics["underline_offset"],
                "overline": ascent * 0.9,
                "line-through": decoration_metrics["strike_offset"],
            }[decoration]
            thickness = (
                decoration_thickness
                or decoration_metrics[
                    "strike_thickness"
                    if decoration == "line-through"
                    else "underline_thickness"
                ]
            )
            _composite_rule(
                field,
                line_x,
                baseline + offset,
                line_width,
                thickness,
                clip_rect=(x, y, x + box_width, y + box_height)
                if overflow == "clip"
                else None,
            )

    tex = ctx.r32f()
    tex.write(field.tobytes())

    from sdf_ui.core.sdf import SDFTexture

    return SDFTexture(tex=tex, context=ctx)


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


def _composite_glyph(field, patch, x, y, scale, distance_scale, clip_rect=None):
    glyph_field = patch["field"]
    source_units_per_pixel = patch["units_per_pixel"]
    source_to_dest = scale * source_units_per_pixel
    distance_to_dest = distance_scale * source_units_per_pixel
    if source_to_dest <= 0:
        return

    left = x + patch["origin"][0] * scale
    top = y + patch["origin"][1] * scale
    right = left + glyph_field.shape[1] * source_to_dest
    bottom = top + glyph_field.shape[0] * source_to_dest

    height, width = field.shape
    x0 = max(0, int(floor(left)))
    y0 = max(0, int(floor(top)))
    x1 = min(width, int(ceil(right)))
    y1 = min(height, int(ceil(bottom)))
    if clip_rect is not None:
        clip_left, clip_bottom, clip_right, clip_top = clip_rect
        x0 = max(x0, int(ceil(clip_left)))
        y0 = max(y0, int(ceil(clip_bottom)))
        x1 = min(x1, int(ceil(clip_right)))
        y1 = min(y1, int(ceil(clip_top)))
    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1]
    source_x = (xx - left) / source_to_dest
    source_y = (yy - top) / source_to_dest
    if patch.get("hinted_size") is not None:
        # The hinted mask already carries pixel-scale coverage. Min-filtering
        # its upsampled distance field would expand thin strokes and close
        # counters, so sample that field bilinearly.
        distances = _sample_bilinear(glyph_field, source_x, source_y)
    else:
        distances = _sample_sdf(glyph_field, source_x, source_y, source_to_dest)
    distances = distances * distance_to_dest
    field[y0:y1, x0:x1] = np.minimum(field[y0:y1, x0:x1], distances)


def _composite_rule(field, x, y, rule_width, thickness, clip_rect=None):
    """Union a rectangular typographic rule into the text SDF."""
    half_width = max(rule_width, 0.0) / 2
    half_thickness = max(thickness, 0.1) / 2
    left, right = x, x + max(rule_width, 0.0)
    top, bottom = y - half_thickness, y + half_thickness
    height, width = field.shape
    x0, x1 = max(0, int(floor(left))), min(width, int(ceil(right)))
    y0, y1 = max(0, int(floor(top))), min(height, int(ceil(bottom)))
    if clip_rect is not None:
        clip_left, clip_bottom, clip_right, clip_top = clip_rect
        x0, x1 = max(x0, int(ceil(clip_left))), min(x1, int(ceil(clip_right)))
        y0, y1 = max(y0, int(ceil(clip_bottom))), min(y1, int(ceil(clip_top)))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    qx = np.abs(xx + 0.5 - (x + half_width)) - half_width
    qy = np.abs(yy + 0.5 - y) - half_thickness
    distance = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(
        np.maximum(qx, qy), 0
    )
    field[y0:y1, x0:x1] = np.minimum(field[y0:y1, x0:x1], distance)


def _sample_sdf(values, x, y, source_to_dest):
    if source_to_dest >= 1.0:
        return _sample_bilinear(values, x, y)

    source_pixels_per_dest = 1.0 / source_to_dest
    radius = max(1, int(ceil(source_pixels_per_dest * 0.25)))
    sampled = _sample_bilinear(values, x, y)
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            if offset_x == 0 and offset_y == 0:
                continue
            sampled = np.minimum(
                sampled, _sample_bilinear(values, x + offset_x, y + offset_y)
            )
    return sampled


def _sample_bilinear(values, x, y):
    height, width = values.shape
    x = np.clip(x, 0.0, width - 1.0)
    y = np.clip(y, 0.0, height - 1.0)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = x - x0
    ty = y - y0

    top = values[y0, x0] * (1.0 - tx) + values[y0, x1] * tx
    bottom = values[y1, x0] * (1.0 - tx) + values[y1, x1] * tx
    return top * (1.0 - ty) + bottom * ty


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
        return {
            "field": np.empty((0, 0), dtype=np.float32),
            "origin": (0.0, 0.0),
            "units_per_pixel": _font_metrics(path)["units_per_em"] / raster_size,
            "hinted_size": hinted_size,
        }

    import cv2

    coverage = np.asarray(mask, dtype=np.uint8).reshape(height, width)
    pad = 2
    alpha = np.pad(coverage, pad, mode="constant").astype(np.float32) / 255.0
    inside = (alpha >= 0.5).astype(np.uint8)
    outside = 1 - inside
    outside_distance = cv2.distanceTransform(
        outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    inside_distance = cv2.distanceTransform(
        inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    # Offset distances to the half-pixel contour. Keep a one-pixel-per-pixel
    # gradient: the fill shader derives its antialias width from this gradient,
    # so adding grayscale coverage here would distort that width.
    field = (
        outside_distance
        - inside_distance
        + inside.astype(np.float32)
        - 0.5
    )
    field = np.flipud(field).astype(np.float32, copy=True)

    font_metrics = _font_metrics(path)
    units_per_pixel = font_metrics["units_per_em"] / raster_size
    pixel_center_offset = 0.5
    pixel_to_font_units = units_per_pixel
    origin = (
        (float(offset[0]) - pad - pixel_center_offset) * pixel_to_font_units,
        (-float(offset[1]) - height - pad - pixel_center_offset)
        * pixel_to_font_units,
    )
    return {
        "field": field,
        "origin": origin,
        "units_per_pixel": units_per_pixel,
        "hinted_size": hinted_size,
    }


@lru_cache(maxsize=1024)
def _cached_glyph_sdf(path, glyph_id, samples, glyph_cache_size, variation_items=()):
    raw_contours = _raw_glyph_contours_by_id(path, glyph_id, samples, variation_items)
    glyph_metrics = _glyph_metrics(path, glyph_id, variation_items)
    if not raw_contours or glyph_metrics["name"] is None:
        return {
            "field": np.empty((0, 0), dtype=np.float32),
            "origin": (0.0, 0.0),
            "units_per_pixel": 1.0,
        }

    font_metrics = _font_metrics(path)
    units_per_pixel = font_metrics["units_per_em"] / glyph_cache_size
    pad = max(2, int(glyph_cache_size * 0.08))
    pad_units = pad * units_per_pixel
    x_min, y_min, x_max, y_max = glyph_metrics["bbox"]
    origin = (x_min - pad_units, y_min - pad_units)
    width = max(1, int(ceil((x_max - x_min) / units_per_pixel)) + pad * 2)
    height = max(1, int(ceil((y_max - y_min) / units_per_pixel)) + pad * 2)

    contours = tuple(
        tuple(
            (
                (point[0] - origin[0]) / units_per_pixel,
                (point[1] - origin[1]) / units_per_pixel,
            )
            for point in contour
        )
        for contour in raw_contours
    )

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
    on_contour = distance_sq <= 1e-6
    internal_contour = np.zeros_like(inside)
    internal_contour[1:-1, 1:-1] = (
        (inside[:-2, 1:-1] & inside[2:, 1:-1])
        | (inside[1:-1, :-2] & inside[1:-1, 2:])
    )
    inside |= on_contour & internal_contour

    import cv2

    inside_mask = inside.astype(np.uint8)
    outside_mask = 1 - inside_mask
    distance_inside = cv2.distanceTransform(
        inside_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    distance_outside = cv2.distanceTransform(
        outside_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    field = distance_outside - distance_inside + inside_mask.astype(np.float32) - 0.5
    return {"field": field, "origin": origin, "units_per_pixel": units_per_pixel}


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
                "samples",
                "cache_size",
                "line_height",
                "oversample",
                "min_render_size",
                "box",
                "align",
                "valign",
                "fallback_fonts",
                "missing_glyph",
                "clip",
                "overflow",
                "trim_trailing_spaces",
                "white_space",
                "first_line_indent",
                "max_lines",
                "decoration",
                "decoration_thickness",
                "word_spacing",
                "letter_spacing",
                "vertical_metrics",
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
                "samples": 16,
                "cache_size": 128,
                "line_height": 1.2,
                "oversample": 2.5,
                "min_render_size": 64,
                "box": None,
                "align": "left",
                "valign": "middle",
                "fallback_fonts": (),
                "missing_glyph": "ignore",
                "clip": False,
                "overflow": "visible",
                "trim_trailing_spaces": True,
                "white_space": "normal",
                "first_line_indent": 0.0,
                "max_lines": None,
                "decoration": (),
                "decoration_thickness": 0.0,
                "word_spacing": 0.0,
                "letter_spacing": 0.0,
                "vertical_metrics": "hhea",
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
