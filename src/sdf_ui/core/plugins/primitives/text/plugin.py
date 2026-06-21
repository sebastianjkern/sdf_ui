__docformat__ = "google"

from functools import lru_cache
from math import ceil, floor

import numpy as np
import ttfquery
from ttfquery import describe

from sdf_ui.core.plugins.base import Plugin, PluginFamily, TextureKind
from sdf_ui.core.plugins.primitives.glyph.plugin import (
    _contour_distance_sq,
    _flatten_contour,
    _points_inside_contour,
    _raw_glyph_contours,
)


def render_text(renderer, _inputs, params):
    ctx = renderer.ctx
    width, height = ctx.size
    font_size = float(params["size"])
    ox = float(params["ox"])
    oy = float(params["oy"])
    path = params["path"]
    samples = int(params["samples"])
    cache_size = int(params["cache_size"])
    line_height = float(params["line_height"])

    metrics = _font_metrics(path)
    scale = font_size / metrics["units_per_em"]
    field = np.full((height, width), max(width, height), dtype=np.float32)

    previous = None
    pen_x = 0.0
    pen_y = 0.0
    line_advance = metrics["units_per_em"] * line_height

    for char in params["value"]:
        if char == "\n":
            previous = None
            pen_x = 0.0
            pen_y += line_advance
            continue

        glyph_metrics = _glyph_metrics(path, char)
        if previous is not None:
            pen_x += _kerning(path, previous, char)

        if glyph_metrics["name"] is not None:
            patch = _cached_glyph_sdf(path, char, samples, cache_size)
            if patch["field"].size:
                _composite_glyph(
                    field,
                    patch,
                    ox + pen_x * scale,
                    oy + pen_y * scale,
                    scale,
                    metrics["units_per_em"] / cache_size,
                )

        pen_x += glyph_metrics["advance"]
        previous = char

    tex = ctx.r32f()
    tex.write(field.tobytes())

    from sdf_ui.core.sdf import SDFTexture

    return SDFTexture(tex=tex, context=ctx)


def _composite_glyph(field, patch, x, y, scale, source_units_per_pixel):
    glyph_field = patch["field"]
    source_to_dest = scale * source_units_per_pixel
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
    distances = _sample_bilinear(glyph_field, source_x, source_y) * source_to_dest
    field[y0:y1, x0:x1] = np.minimum(field[y0:y1, x0:x1], distances)


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


@lru_cache(maxsize=512)
def _cached_glyph_sdf(path, char, samples, cache_size):
    raw_contours = _raw_glyph_contours(path, char)
    glyph_metrics = _glyph_metrics(path, char)
    if not raw_contours or glyph_metrics["name"] is None:
        return {"field": np.empty((0, 0), dtype=np.float32), "origin": (0.0, 0.0)}

    font_metrics = _font_metrics(path)
    units_per_pixel = font_metrics["units_per_em"] / cache_size
    pad = max(2, int(cache_size * 0.08))
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
            for point in _flatten_contour(contour, samples)
        )
        for contour in raw_contours
    )

    yy, xx = np.mgrid[0:height, 0:width]
    points = np.stack((xx.astype(np.float32), yy.astype(np.float32)), axis=-1)
    distance_sq = np.full((height, width), np.inf, dtype=np.float32)
    inside = np.zeros((height, width), dtype=bool)

    for contour in contours:
        distance_sq = np.minimum(distance_sq, _contour_distance_sq(points, contour))
        inside ^= _points_inside_contour(points, contour)

    field = np.sqrt(distance_sq).astype(np.float32)
    field[inside] *= -1.0
    return {"field": field, "origin": origin}


@lru_cache(maxsize=64)
def _font_metrics(path):
    font = describe.openFont(path)
    return {
        "units_per_em": float(font["head"].unitsPerEm),
        "ascent": float(font["hhea"].ascent),
        "descent": float(font["hhea"].descent),
    }


@lru_cache(maxsize=512)
def _glyph_metrics(path, char):
    font = describe.openFont(path)
    glyph_name = ttfquery.glyphquery.glyphName(font, char)
    if glyph_name is None:
        return {"name": None, "advance": 0.0, "bbox": (0.0, 0.0, 0.0, 0.0)}

    advance, _left_side_bearing = font["hmtx"].metrics.get(glyph_name, (0, 0))
    glyph = font["glyf"][glyph_name]
    if glyph.numberOfContours == 0:
        bbox = (0.0, 0.0, 0.0, 0.0)
    else:
        bbox = (
            float(glyph.xMin),
            float(glyph.yMin),
            float(glyph.xMax),
            float(glyph.yMax),
        )
    return {"name": glyph_name, "advance": float(advance), "bbox": bbox}


@lru_cache(maxsize=2048)
def _kerning(path, left, right):
    font = describe.openFont(path)
    left_name = ttfquery.glyphquery.glyphName(font, left)
    right_name = ttfquery.glyphquery.glyphName(font, right)
    if left_name is None or right_name is None or "kern" not in font:
        return 0.0

    kern_table = font["kern"]
    for table in getattr(kern_table, "kernTables", ()):
        value = getattr(table, "kernTable", {}).get((left_name, right_name))
        if value is not None:
            return float(value)
    return 0.0


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
            ),
            defaults={
                "size": 64,
                "ox": 0,
                "oy": 0,
                "path": "fonts/georgia_regular.ttf",
                "samples": 16,
                "cache_size": 128,
                "line_height": 1.2,
            },
            public=True,
            render_func=render_text,
        )
    )
