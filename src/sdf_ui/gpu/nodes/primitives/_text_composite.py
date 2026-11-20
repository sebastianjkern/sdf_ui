"""SDF image composition helpers used by text glyph rendering."""

from math import ceil, floor

import numpy as np

_FIELD_KEY = "field"
_UNITS_PER_PIXEL_KEY = "units_per_pixel"
_ORIGIN_KEY = "origin"

def _composite_glyph(field, patch, x, y, scale, distance_scale, clip_rect=None):
    glyph_field = patch[_FIELD_KEY]
    source_units_per_pixel = patch[_UNITS_PER_PIXEL_KEY]
    source_to_dest = scale * source_units_per_pixel
    distance_to_dest = distance_scale * source_units_per_pixel
    if source_to_dest <= 0:
        return

    left = x + patch[_ORIGIN_KEY][0] * scale
    top = y + patch[_ORIGIN_KEY][1] * scale
    right = left + glyph_field.shape[1] * source_to_dest
    bottom = top + glyph_field.shape[0] * source_to_dest

    bounds = _composite_bounds(field.shape, left, top, right, bottom, clip_rect)
    if bounds is None:
        return
    x0, y0, x1, y1 = bounds
    yy, xx = np.mgrid[y0:y1, x0:x1]
    source_x = (xx - left) / source_to_dest
    source_y = (yy - top) / source_to_dest
    distances = _sample_glyph_patch(patch, source_x, source_y, source_to_dest)
    distances = distances * distance_to_dest
    field[y0:y1, x0:x1] = np.minimum(field[y0:y1, x0:x1], distances)


def _composite_bounds(shape, left, top, right, bottom, clip_rect):
    height, width = shape
    bounds = [max(0, int(floor(left))), max(0, int(floor(top))),
              min(width, int(ceil(right))), min(height, int(ceil(bottom)))]
    if clip_rect is not None:
        clip_left, clip_bottom, clip_right, clip_top = clip_rect
        bounds[0] = max(bounds[0], int(ceil(clip_left)))
        bounds[1] = max(bounds[1], int(ceil(clip_bottom)))
        bounds[2] = min(bounds[2], int(ceil(clip_right)))
        bounds[3] = min(bounds[3], int(ceil(clip_top)))
    return tuple(bounds) if bounds[0] < bounds[2] and bounds[1] < bounds[3] else None


def _sample_glyph_patch(patch, source_x, source_y, source_to_dest):
    if patch.get("hinted_size") is None:
        return _sample_sdf(patch[_FIELD_KEY], source_x, source_y, source_to_dest)
    return _sample_bilinear(patch[_FIELD_KEY], source_x, source_y)


def _composite_rule(field, x, y, rule_width, thickness, clip_rect=None):
    """Union a rectangular typographic rule into the text SDF."""
    half_width = max(rule_width, 0.0) / 2
    half_thickness = max(thickness, 0.1) / 2
    left, right = x, x + max(rule_width, 0.0)
    top, bottom = y - half_thickness, y + half_thickness
    bounds = _composite_bounds(field.shape, left, top, right, bottom, clip_rect)
    if bounds is None:
        return
    x0, y0, x1, y1 = bounds
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = _rectangle_rule_distance(xx, yy, x, y, half_width, half_thickness)
    field[y0:y1, x0:x1] = np.minimum(field[y0:y1, x0:x1], distance)


def _rectangle_rule_distance(xx, yy, x, y, half_width, half_thickness):
    qx = np.abs(xx + 0.5 - (x + half_width)) - half_width
    qy = np.abs(yy + 0.5 - y) - half_thickness
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return outside + np.minimum(np.maximum(qx, qy), 0)


def _sample_sdf(values, x, y, source_to_dest):
    if source_to_dest >= 1.0:
        return _sample_bilinear(values, x, y)

    half_pixel = 0.5 / source_to_dest
    radius = ceil(half_pixel)
    sampled = _sample_bilinear(values, x, y)
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            if offset_x == 0 and offset_y == 0:
                continue
            dx = min(max(offset_x, -half_pixel), half_pixel)
            dy = min(max(offset_y, -half_pixel), half_pixel)
            sampled = np.minimum(
                sampled, _sample_bilinear(values, x + dx, y + dy)
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
