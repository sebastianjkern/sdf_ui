from sdf_ui.fonts import default_font_path

__docformat__ = "google"

from functools import lru_cache
from math import ceil

import numpy as np
import ttfquery
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from ttfquery import describe, glyph

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind


def render_glyph(renderer, _inputs, params):
    ctx = renderer.ctx
    width, height = ctx.size
    contours = _flatten_glyph(
        params["path"] or default_font_path(),
        params["char"],
        float(params["scale"]),
        float(params["ox"]),
        float(params["oy"]),
        int(params["samples"]),
    )

    field = np.empty((height, width), dtype=np.float32)
    if not contours:
        field.fill(max(width, height))
    else:
        yy, xx = np.mgrid[0:height, 0:width]
        points = np.stack((xx.astype(np.float32), yy.astype(np.float32)), axis=-1)
        distance_sq = np.full((height, width), np.inf, dtype=np.float32)
        winding = np.zeros((height, width), dtype=np.int16)

        for contour in contours:
            distance_sq = np.minimum(distance_sq, _contour_distance_sq(points, contour))
            winding += _contour_winding(points, contour)

        field = np.sqrt(distance_sq).astype(np.float32)
        field[winding != 0] *= -1.0

    tex = ctx.r32f()
    tex.write(field.tobytes())

    from sdf_ui.core.sdf import SDFTexture

    return SDFTexture(tex=tex, context=ctx)


@lru_cache(maxsize=256)
def _flatten_glyph(path, char, scale, ox, oy, samples):
    raw_contours = _raw_glyph_contours(path, char)
    return tuple(
        tuple(
            _transform_point(point, scale, ox, oy)
            for point in _flatten_contour(contour, samples)
        )
        for contour in raw_contours
    )


@lru_cache(maxsize=64)
def _raw_glyph_contours(font_file_path, char):
    font = describe.openFont(font_file_path)
    glyph_name = ttfquery.glyphquery.glyphName(font, char)
    if glyph_name is None:
        return ()
    return tuple(
        tuple(contour) for contour in glyph.Glyph(glyph_name).calculateContours(font)
    )


class _SampledOutlinePen(BasePen):
    """Decompose a font glyph into sampled line contours, including CFF curves."""

    def __init__(self, glyph_set, samples):
        super().__init__(glyph_set)
        self.samples = max(1, int(samples))
        self.contours = []
        self.current = None

    def _moveTo(self, point):
        if self.current:
            self._finish()
        self.current = [tuple(map(float, point))]

    def _lineTo(self, point):
        self.current.append(tuple(map(float, point)))

    def _curveToOne(self, control1, control2, end):
        start = self._getCurrentPoint()
        for index in range(1, self.samples + 1):
            t = index / self.samples
            u = 1.0 - t
            self.current.append(
                (
                    u**3 * start[0]
                    + 3 * u**2 * t * control1[0]
                    + 3 * u * t**2 * control2[0]
                    + t**3 * end[0],
                    u**3 * start[1]
                    + 3 * u**2 * t * control1[1]
                    + 3 * u * t**2 * control2[1]
                    + t**3 * end[1],
                )
            )

    def _qCurveToOne(self, control, end):
        start = self._getCurrentPoint()
        for index in range(1, self.samples + 1):
            t = index / self.samples
            u = 1.0 - t
            self.current.append(
                (
                    u * u * start[0] + 2 * u * t * control[0] + t * t * end[0],
                    u * u * start[1] + 2 * u * t * control[1] + t * t * end[1],
                )
            )

    def _closePath(self):
        self._finish(close=True)

    def _endPath(self):
        self._finish(close=True)

    def _finish(self, close=False):
        if self.current:
            if close and self.current[0] != self.current[-1]:
                self.current.append(self.current[0])
            self.contours.append(tuple(self.current))
        self.current = None


@lru_cache(maxsize=64)
def _outline_font(font_file_path):
    return TTFont(font_file_path, lazy=False)


@lru_cache(maxsize=128)
def _raw_glyph_contours_by_id(font_file_path, glyph_id, samples, variation_items=()):
    font = _outline_font(font_file_path)
    order = font.getGlyphOrder()
    if glyph_id < 0 or glyph_id >= len(order):
        return ()
    location = dict(variation_items) if variation_items else None
    glyph_set = font.getGlyphSet(location=location)
    pen = _SampledOutlinePen(glyph_set, samples)
    glyph_set[order[glyph_id]].draw(pen)
    pen._finish(close=True)
    return tuple(contour for contour in pen.contours if len(contour) > 1)


def _flatten_contour(contour, samples):
    points = _expanded_contour(contour)
    if not points:
        return ()

    flattened = [points[0][0]]
    current = points[0][0]
    index = 1
    while index < len(points):
        point, on_curve = points[index]
        if on_curve:
            flattened.append(point)
            current = point
            index += 1
            continue

        end, end_on_curve = points[index + 1]
        if not end_on_curve:
            raise ValueError(
                "Expanded glyph contour contains adjacent off-curve points"
            )
        flattened.extend(_sample_quadratic(current, point, end, samples))
        current = end
        index += 2

    if flattened[0] != flattened[-1]:
        flattened.append(flattened[0])
    return flattened


def _expanded_contour(contour):
    points = list(contour)
    if len(points) > 1 and points[0][0] == points[-1][0]:
        points.pop()
    if not points:
        return ()

    expanded = []
    for index, item in enumerate(points):
        point, on_curve = item
        expanded.append((tuple(point), bool(on_curve)))
        next_point, next_on_curve = points[(index + 1) % len(points)]
        if not on_curve and not next_on_curve:
            expanded.append(
                (_middle(point[0], point[1], next_point[0], next_point[1]), True)
            )

    if not expanded[0][1]:
        previous = expanded[-1][0]
        first = expanded[0][0]
        expanded.insert(
            0, (_middle(previous[0], previous[1], first[0], first[1]), True)
        )

    first = expanded[0]
    expanded.append(first)
    return tuple(expanded)


def _sample_quadratic(a, b, c, samples):
    count = max(1, ceil(samples))
    result = []
    ax, ay = a
    bx, by = b
    cx, cy = c
    for index in range(1, count + 1):
        t = index / count
        u = 1.0 - t
        result.append(
            (
                u * u * ax + 2.0 * u * t * bx + t * t * cx,
                u * u * ay + 2.0 * u * t * by + t * t * cy,
            )
        )
    return result


def _middle(x1, y1, x2, y2):
    return x1 + 0.5 * (x2 - x1), y1 + 0.5 * (y2 - y1)


def _transform_point(point, scale, ox, oy):
    return point[0] * scale + ox, point[1] * scale + oy


def _contour_distance_sq(points, contour):
    distance_sq = np.full(points.shape[:2], np.inf, dtype=np.float32)
    for start, end in zip(contour, contour[1:]):
        ax, ay = start
        bx, by = end
        segment = np.array((bx - ax, by - ay), dtype=np.float32)
        length_sq = float(np.dot(segment, segment))
        if length_sq == 0.0:
            continue
        relative = points - np.array((ax, ay), dtype=np.float32)
        t = np.clip(
            (relative[..., 0] * segment[0] + relative[..., 1] * segment[1]) / length_sq,
            0.0,
            1.0,
        )
        closest_x = ax + t * segment[0]
        closest_y = ay + t * segment[1]
        dx = points[..., 0] - closest_x
        dy = points[..., 1] - closest_y
        distance_sq = np.minimum(distance_sq, dx * dx + dy * dy)
    return distance_sq


def _contour_winding(points, contour):
    x = points[..., 0]
    y = points[..., 1]
    winding = np.zeros(points.shape[:2], dtype=np.int16)
    for start, end in zip(contour, contour[1:]):
        x0, y0 = start
        x1, y1 = end
        is_left = (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)
        upward = (y0 <= y) & (y1 > y) & (is_left > 0)
        downward = (y0 > y) & (y1 <= y) & (is_left < 0)
        winding += upward.astype(np.int16)
        winding -= downward.astype(np.int16)
    return winding


def _points_inside_contour(points, contour):
    return _contour_winding(points, contour) != 0


def register_plugins(registry):
    registry.register(
        Plugin(
            "glyph",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("char", "scale", "ox", "oy", "path", "samples"),
            defaults={"path": None, "samples": 16},
            render_func=render_glyph,
        )
    )
