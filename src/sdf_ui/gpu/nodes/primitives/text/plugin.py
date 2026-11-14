__docformat__ = "google"

from functools import lru_cache
from math import ceil, floor
from numbers import Real

import numpy as np

from sdf_ui.fonts import (
    font_metrics as _font_metrics,
)
from sdf_ui.fonts import (
    font_paths,
    layout_font_paths,
    layout_text,
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
    samples = int(params["samples"])
    cache_size = int(params["cache_size"])
    line_height = float(params["line_height"])
    oversample = float(params["oversample"])
    min_render_size = float(params["min_render_size"])
    box = _normalize_box(params["box"])
    align = params["align"]
    valign = params["valign"]
    missing_glyph = params["missing_glyph"]
    shape_options = {
        key: params[key]
        for key in ("direction", "script", "language", "features", "variations")
    }
    for key in ("features", "variations"):
        if isinstance(shape_options[key], tuple):
            shape_options[key] = dict(shape_options[key])

    if not np.isfinite(font_size) or font_size <= 0:
        raise ValueError("text size must be a finite positive number")
    if not np.isfinite(line_height) or line_height <= 0:
        raise ValueError("text line_height must be a finite positive number")

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
        **shape_options,
    )
    ascent, descent = text_vertical_metrics(layout_font_paths(paths, lines), font_size)

    if align not in {"left", "center", "right"}:
        raise ValueError("text align must be 'left', 'center', or 'right'")
    if valign not in {"top", "middle", "bottom"}:
        raise ValueError("text valign must be 'top', 'middle', or 'bottom'")
    if box is None and align != "left":
        raise ValueError("text alignment requires a box")

    x, y, box_width, box_height = box if box is not None else (ox, oy, None, None)
    if box is None:
        first_baseline = oy
    else:
        block_height = ascent - descent + max(len(lines) - 1, 0) * line_advance
        if valign == "top":
            first_baseline = y + box_height - ascent
        elif valign == "middle":
            block_bottom = y + (box_height - block_height) / 2
            first_baseline = (
                block_bottom - descent + max(len(lines) - 1, 0) * line_advance
            )
        else:
            first_baseline = y - descent + max(len(lines) - 1, 0) * line_advance

    for line_index, (placements, line_width, _line_text) in enumerate(lines):
        if box is None:
            line_x = ox
        elif align == "left":
            line_x = x
        elif align == "center":
            line_x = x + (box_width - line_width) / 2
        else:
            line_x = x + box_width - line_width

        baseline = first_baseline - line_index * line_advance
        for glyph_id, glyph_path, glyph_x, glyph_y in placements:
            glyph_metrics = _font_metrics(glyph_path)
            glyph_scale = font_size / glyph_metrics["units_per_em"]
            glyph_distance_scale = effective_font_size / glyph_metrics["units_per_em"]
            patch = _cached_glyph_sdf(
                glyph_path,
                glyph_id,
                samples,
                glyph_cache_size,
                tuple(sorted((shape_options["variations"] or {}).items())),
            )
            if patch["field"].size:
                _composite_glyph(
                    field,
                    patch,
                    line_x + glyph_x,
                    baseline + glyph_y,
                    glyph_scale,
                    glyph_distance_scale,
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


def _composite_glyph(field, patch, x, y, scale, distance_scale):
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
    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1]
    source_x = (xx - left) / source_to_dest
    source_y = (yy - top) / source_to_dest
    distances = _sample_sdf(glyph_field, source_x, source_y, source_to_dest)
    distances = distances * distance_to_dest
    field[y0:y1, x0:x1] = np.minimum(field[y0:y1, x0:x1], distances)


def _sample_sdf(values, x, y, source_to_dest):
    if source_to_dest >= 1.0:
        return _sample_bilinear(values, x, y)

    source_pixels_per_dest = 1.0 / source_to_dest
    radius = max(1, int(ceil(source_pixels_per_dest * 0.5)))
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

    field = np.sqrt(distance_sq).astype(np.float32)
    field[winding != 0] *= -1.0
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
                "oversample": 2.0,
                "min_render_size": 64,
                "box": None,
                "align": "left",
                "valign": "middle",
                "fallback_fonts": (),
                "missing_glyph": "ignore",
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
