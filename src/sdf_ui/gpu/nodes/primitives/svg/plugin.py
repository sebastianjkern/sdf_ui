"""A small SVG path renderer designed to ingest MathJax SVG output."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.svgLib.path import parse_path

from sdf_ui.gpu.nodes.base import Plugin, PluginFamily, TextureKind
from sdf_ui.gpu.nodes.primitives.glyph.plugin import (
    _contour_distance_sq,
    _contour_winding,
    _point_line_distance,
)

_XLINK_NS = "http://www.w3.org/1999/xlink"
_TRANSFORM_RE = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")


def _local_name(tag):
    return tag.rsplit("}", 1)[-1]


def _numbers(value):
    return tuple(float(item) for item in _NUMBER_RE.findall(value or ""))


def _multiply(first, second):
    """Multiply SVG affine matrices represented as (a,b,c,d,e,f)."""
    a, b, c, d, e, f = first
    g, h, i, j, k, ell = second
    return (
        a * g + c * h,
        b * g + d * h,
        a * i + c * j,
        b * i + d * j,
        a * k + c * ell + e,
        b * k + d * ell + f,
    )


_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
_FLATTENING_TOLERANCE_FACTOR = 0.25
_CUBIC_TOLERANCE_DIVISOR = 0.25
_NONZERO_FILL = "nonzero"
_SVG_SKIP_ELEMENTS = {"defs", "title", "desc", "metadata", "style"}
_SVG_UNSUPPORTED_CONTAINERS = {"symbol", "clipPath", "mask", "pattern"}


def _parse_transform(value):
    result = _IDENTITY
    for name, arguments in _TRANSFORM_RE.findall(value or ""):
        values = _numbers(arguments)
        current = _SVGTransform.matrix(name, values, arguments)
        result = _multiply(result, current)
    return result


class _SVGTransform:
    @staticmethod
    def matrix(name, values, arguments):
        parsers = {
            "matrix": _SVGTransform.matrix_values,
            "translate": _SVGTransform.translate,
            "scale": _SVGTransform.scale,
            "rotate": _SVGTransform.rotate,
            "skewX": _SVGTransform.skew_x,
            "skewY": _SVGTransform.skew_y,
        }
        parser = parsers.get(name)
        result = None if parser is None else parser(values)
        if result is None:
            raise ValueError(f"unsupported SVG transform: {name}({arguments})")
        return result

    @staticmethod
    def matrix_values(values):
        return values if len(values) == 6 else None

    @staticmethod
    def translate(values):
        if len(values) not in {1, 2}:
            return None
        y = values[1] if len(values) == 2 else 0
        return 1, 0, 0, 1, values[0], y

    @staticmethod
    def scale(values):
        if len(values) not in {1, 2}:
            return None
        return values[0], 0, 0, values[-1], 0, 0

    @staticmethod
    def rotate(values):
        if len(values) not in {1, 3}:
            return None
        angle = math.radians(values[0])
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = (cosine, sine, -sine, cosine, 0, 0)
        if len(values) != 3:
            return rotation
        x, y = values[1:]
        return _multiply(_multiply((1, 0, 0, 1, x, y), rotation), (1, 0, 0, 1, -x, -y))

    @staticmethod
    def skew_x(values):
        if len(values) != 1:
            return None
        tangent = math.tan(math.radians(values[0]))
        return 1, 0, tangent, 1, 0, 0

    @staticmethod
    def skew_y(values):
        if len(values) != 1:
            return None
        tangent = math.tan(math.radians(values[0]))
        return 1, tangent, 0, 1, 0, 0


def _transform_point(matrix, point):
    a, b, c, d, e, f = matrix
    x, y = point
    return a * x + c * y + e, b * x + d * y + f


def _style(element, inherited):
    result = dict(inherited)
    style_text = element.attrib.get("style", "")
    inline = {}
    for declaration in style_text.split(";"):
        if ":" in declaration:
            key, value = declaration.split(":", 1)
            inline[key.strip()] = value.strip()
    for key in (
        "fill", "fill-rule", "stroke", "stroke-width", "opacity", "display",
        "visibility", "font-family", "font-size", "font-style", "font-weight",
        "text-anchor",
    ):
        value = element.attrib.get(key, inline.get(key))
        if value is not None:
            result[key] = value
    return result


class _FlatteningPen(BasePen):
    def __init__(self, matrix, quality):
        super().__init__(None)
        self.matrix = matrix
        self.tolerance = (16 / max(1, quality)) ** 2 * _CUBIC_TOLERANCE_DIVISOR
        self.contours = []
        self.current = None

    def _moveTo(self, point):
        if self.current:
            self._finish(close=True)
        self.current = [_transform_point(self.matrix, point)]

    def _lineTo(self, point):
        self.current.append(_transform_point(self.matrix, point))

    def _curveToOne(self, control1, control2, end):
        start = _transform_point(self.matrix, self._getCurrentPoint())
        self.current.extend(_flatten_cubic(
            start,
            _transform_point(self.matrix, control1),
            _transform_point(self.matrix, control2),
            _transform_point(self.matrix, end),
            self.tolerance,
        ))

    def _qCurveToOne(self, control, end):
        self.current.extend(
            _flatten_quadratic(
                _transform_point(self.matrix, self._getCurrentPoint()),
                _transform_point(self.matrix, control),
                _transform_point(self.matrix, end),
                self.tolerance,
            )
        )

    def _closePath(self):
        self._finish(close=True)

    def _endPath(self):
        # SVG fill implicitly closes open subpaths.
        self._finish(close=True)

    def _finish(self, close):
        if self.current:
            if close and len(self.current) > 1 and self.current[0] != self.current[-1]:
                self.current.append(self.current[0])
            if len(self.current) > 1:
                self.contours.append(tuple(self.current))
        self.current = None


def _parse_svg(source):
    source = _read_svg_source(source)
    _validate_svg_source(source)
    root = _parse_svg_xml(source)
    if _local_name(root.tag) != "svg":
        raise ValueError("SVG source root element must be <svg>")
    view_box = _numbers(root.attrib.get("viewBox", root.attrib.get("viewbox", "")))
    if len(view_box) != 4 or view_box[2] <= 0 or view_box[3] <= 0:
        raise ValueError("SVG root must define a positive four-number viewBox")
    return root, view_box


def _read_svg_source(source):
    if isinstance(source, Path):
        try:
            source = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"unable to read SVG file: {source}") from exc
    if not isinstance(source, str):
        raise TypeError("SVG source must be markup text or a pathlib.Path")
    return source


def _validate_svg_source(source):
    if len(source) > 16_000_000:
        raise ValueError("SVG source exceeds the 16 MB size limit")
    if "<!DOCTYPE" in source.upper() or "<!ENTITY" in source.upper():
        raise ValueError("SVG document types and entity declarations are not supported")


def _parse_svg_xml(source):
    try:
        return ET.fromstring(source)
    except ET.ParseError as exc:
        raise ValueError(f"invalid SVG XML: {exc}") from exc


@lru_cache(maxsize=1024)
def _svg_path_commands(path_data):
    recording = RecordingPen()
    try:
        parse_path(path_data, recording)
    except (ValueError, AssertionError) as exc:
        raise ValueError(f"invalid or unsupported SVG path: {exc}") from exc
    return tuple(recording.value)


class _SVGText:
    @staticmethod
    def shapes(element, matrix, style, samples, view_box_height, target_height):
        from sdf_ui.fonts import font_metrics, layout_text
        from sdf_ui.gpu.nodes.primitives.glyph.plugin import _raw_glyph_contours_by_id

        value = _SVGText._content(element)
        if not value:
            return ()
        font_size = _SVGText._font_size(style)
        font_path = _SVGText._font_path(style)
        lines = layout_text(value, (font_path,), font_size, white_space="pre")
        origin_x, origin_y = _SVGText._origin(element, style, lines)
        fill_rule = style.get("fill-rule", _NONZERO_FILL)
        shapes = []
        for placements, *_rest in lines:
            for glyph in placements:
                positioned = _SVGText._position_glyph(
                    glyph, matrix, origin_x, origin_y, font_size, samples,
                    view_box_height, target_height, font_metrics,
                    _raw_glyph_contours_by_id,
                )
                if positioned:
                    shapes.append((positioned, fill_rule))
        return tuple(shapes)

    @staticmethod
    def _content(element):
        if list(element):
            raise ValueError("SVG <text> with nested <tspan> elements is not supported")
        value = "".join(element.itertext())
        if "\n" in value or "\r" in value:
            raise ValueError("multi-line SVG <text> is not supported")
        return value

    @staticmethod
    def _font_size(style):
        size_text = style.get("font-size", "16").strip()
        match = re.fullmatch(r"([-+0-9.eE]+)(?:px)?", size_text)
        if match is None:
            raise ValueError(f"unsupported SVG font-size '{size_text}'")
        size = float(match.group(1))
        if not math.isfinite(size) or size <= 0:
            raise ValueError("SVG font-size must be finite and positive")
        return size

    @staticmethod
    def _font_path(style):
        from sdf_ui.fonts import default_font_path, system_font

        family = style.get("font-family", "sans-serif").split(",")[0].strip().strip("\"'")
        weight = style.get("font-weight", "normal").lower()
        bold = weight in {"bold", "600", "700", "800", "900"}
        suffix = "Bold" if bold else "Regular"
        if style.get("font-style", "normal").lower() in {"italic", "oblique"}:
            suffix = "Bold Italic" if bold else "Italic"
        try:
            return system_font(family or "sans-serif", suffix)
        except (RuntimeError, LookupError, ValueError):
            return default_font_path()

    @staticmethod
    def _origin(element, style, lines):
        x = _SVGText._first_number(element, "x") + _SVGText._first_number(element, "dx")
        y = _SVGText._first_number(element, "y") + _SVGText._first_number(element, "dy")
        line_width = lines[0][1] if lines else 0.0
        anchor = style.get("text-anchor", "start")
        if anchor == "middle":
            x -= line_width / 2
        elif anchor == "end":
            x -= line_width
        return x, y

    @staticmethod
    def _first_number(element, name):
        values = _numbers(element.attrib.get(name, "0"))
        return values[0] if values else 0.0

    @staticmethod
    def _position_glyph(glyph, matrix, origin_x, origin_y, font_size, samples,
                        view_box_height, target_height, font_metrics, contours_for):
        glyph_id, glyph_path, glyph_x, glyph_y, _cluster = glyph
        metrics = font_metrics(glyph_path)
        glyph_scale = font_size / metrics["units_per_em"]
        pixels_per_unit = target_height / view_box_height * glyph_scale
        quality = max(samples, math.ceil(math.sqrt(1000 * pixels_per_unit / _FLATTENING_TOLERANCE_FACTOR)))
        contours = contours_for(glyph_path, glyph_id, quality, ())
        return tuple(
            tuple(_transform_point(matrix, (
                origin_x + glyph_x + point[0] * glyph_scale,
                origin_y - glyph_y - point[1] * glyph_scale,
            )) for point in contour)
            for contour in contours
        )


def _flatten_quadratic(start, control, end, tolerance, depth=0):
    if depth >= 16 or _point_line_distance(control, start, end) <= tolerance:
        return (end,)
    middle = (
        (start[0] + 2 * control[0] + end[0]) / 4,
        (start[1] + 2 * control[1] + end[1]) / 4,
    )
    left = ((start[0] + control[0]) / 2, (start[1] + control[1]) / 2)
    right = ((control[0] + end[0]) / 2, (control[1] + end[1]) / 2)
    return _flatten_quadratic(
        start, left, middle, tolerance, depth + 1
    ) + _flatten_quadratic(middle, right, end, tolerance, depth + 1)


def _flatten_cubic(start, control1, control2, end, tolerance, depth=0):
    flatness = max(
        _point_line_distance(control1, start, end),
        _point_line_distance(control2, start, end),
    )
    if depth >= 16 or flatness <= tolerance:
        return (end,)
    def middle(a, b):
        return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

    a = middle(start, control1)
    b = middle(control1, control2)
    c = middle(control2, end)
    d = middle(a, b)
    e = middle(b, c)
    f = middle(d, e)
    return _flatten_cubic(
        start, a, d, f, tolerance, depth + 1
    ) + _flatten_cubic(f, e, c, end, tolerance, depth + 1)


def _collect_shapes(root, view_box, center, target_height, samples):
    min_x, min_y, box_width, box_height = view_box
    scale = target_height / box_height
    cx, cy = center
    root_matrix = (
        scale,
        0.0,
        0.0,
        -scale,
        cx - (min_x + box_width / 2) * scale,
        cy + (min_y + box_height / 2) * scale,
    )
    definitions = {}
    for element in root.iter():
        identifier = element.attrib.get("id")
        if identifier:
            definitions[identifier] = element

    shapes = []
    inherited = {
        "fill": "black",
        "fill-rule": _NONZERO_FILL,
        "stroke": "none",
        "stroke-width": "0",
        "opacity": "1",
        "display": "inline",
        "visibility": "visible",
    }

    def skip_element(name, from_use):
        return (name in _SVG_SKIP_ELEMENTS and not from_use) or name in _SVG_UNSUPPORTED_CONTAINERS

    def fill_is_visible(style):
        fill = style.get("fill", "black").lower()
        return fill not in {"none", "transparent"} and float(style.get("opacity", "1")) != 0

    def validate_stroke(style):
        stroke = style.get("stroke", "none")
        stroke_width = float(style.get("stroke-width", "0").replace("px", ""))
        if stroke not in {"none", "transparent"} and stroke_width > 0:
            raise ValueError("SVG strokes are not supported; MathJax fill paths are supported")

    def visit_use(element, matrix, current_style, use_stack):
        href = element.attrib.get("href") or element.attrib.get(f"{{{_XLINK_NS}}}href")
        if not href or not href.startswith("#"):
            raise ValueError("SVG <use> must reference a local #id")
        identifier = href[1:]
        if identifier in use_stack:
            raise ValueError("cyclic SVG <use> reference")
        target = definitions.get(identifier)
        if target is None:
            raise ValueError(f"SVG <use> references unknown id '{identifier}'")
        offset = (float(element.attrib.get("x", 0)), float(element.attrib.get("y", 0)))
        use_matrix = _multiply(matrix, (1, 0, 0, 1, *offset))
        visit(target, use_matrix, current_style, (*use_stack, identifier), True)

    def visit_path(element, matrix, current_style):
        if not fill_is_visible(current_style):
            return
        validate_stroke(current_style)
        path_data = element.attrib.get("d", "")
        if not path_data:
            return
        pen = _FlatteningPen(_multiply(root_matrix, matrix), samples)
        for operator, arguments in _svg_path_commands(path_data):
            getattr(pen, operator)(*arguments)
        pen._finish(close=True)
        if not pen.contours:
            return
        fill_rule = current_style.get("fill-rule", _NONZERO_FILL).lower()
        if fill_rule not in {_NONZERO_FILL, "evenodd"}:
            raise ValueError(f"unsupported SVG fill-rule '{fill_rule}'")
        shapes.append((tuple(pen.contours), fill_rule))

    def visit_text(element, matrix, current_style):
        if fill_is_visible(current_style):
            shapes.extend(_SVGText.shapes(
                element, _multiply(root_matrix, matrix), current_style,
                samples, box_height, target_height,
            ))

    def visit(element, parent_matrix, parent_style, use_stack=(), from_use=False):
        name = _local_name(element.tag)
        if skip_element(name, from_use):
            return
        current_style = _style(element, parent_style)
        if current_style.get("display") == "none" or current_style.get("visibility") == "hidden":
            return
        matrix = _multiply(parent_matrix, _parse_transform(element.attrib.get("transform", "")))
        if name == "use":
            visit_use(element, matrix, current_style, use_stack)
            return
        if name == "path":
            visit_path(element, matrix, current_style)
            return
        if name == "text":
            visit_text(element, matrix, current_style)
            return
        for child in element:
            visit(child, matrix, current_style, use_stack, from_use)

    visit(root, _IDENTITY, inherited)
    if not shapes:
        raise ValueError("SVG contains no filled path geometry")
    return tuple(shapes)


def render_svg(renderer, _inputs, params):
    from sdf_ui.core.sdf import SDFTexture

    ctx = renderer.ctx
    center, target_height, inflate, samples, oversample = _SVGRasterizer.parameters(ctx, params)
    root, view_box = _parse_svg(params["source"])
    curve_samples = max(1, round(samples * math.sqrt(oversample)))
    shapes = _collect_shapes(root, view_box, center, target_height, curve_samples)
    field = _SVGRasterizer.render(ctx.size, oversample, shapes, inflate)
    tex = ctx.r32f()
    tex.write(field.tobytes())
    return SDFTexture(tex=tex, context=ctx)


class _SVGRasterizer:
    @staticmethod
    def parameters(ctx, params):
        center = tuple(float(value) for value in params["center"])
        target_height = float(params["height"])
        samples = max(1, round(int(params["samples"]) * getattr(ctx, "device_scale", 1.0)))
        oversample = params["oversample"]
        if isinstance(oversample, bool) or not isinstance(oversample, (int, np.integer)) or not 1 <= oversample <= 4:
            raise ValueError("SVG oversample must be an integer from 1 to 4")
        oversample = int(oversample)
        if len(center) != 2 or not all(math.isfinite(value) for value in center):
            raise ValueError("SVG center must contain two finite numbers")
        if not math.isfinite(target_height) or target_height <= 0:
            raise ValueError("SVG height must be a finite positive number")
        inflate = float(params["inflate"])
        if not math.isfinite(inflate):
            raise ValueError("SVG inflate must be finite")
        return center, target_height, inflate, samples, oversample

    @staticmethod
    def render(size, oversample, shapes, inflate):
        width, height = size
        high_width = width * oversample
        x_coords = (np.arange(high_width, dtype=np.float32) + 0.5) / oversample - 0.5
        field = np.empty((height, width), dtype=np.float32)
        rows_per_tile = max(1, min(height, 1_000_000 // high_width // oversample))
        for row_start in range(0, height, rows_per_tile):
            row_end = min(height, row_start + rows_per_tile)
            high_y_start, high_y_end = row_start * oversample, row_end * oversample
            y_coords = (np.arange(high_y_start, high_y_end, dtype=np.float32) + 0.5) / oversample - 0.5
            yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
            points = np.stack((xx, yy), axis=-1)
            high_field = np.full(points.shape[:2], np.inf, dtype=np.float32)
            for contours, fill_rule in shapes:
                high_field = np.minimum(high_field, _SVGRasterizer.shape(points, contours, fill_rule))
            high_field -= inflate
            field[row_start:row_end] = high_field.reshape(
                row_end - row_start, oversample, width, oversample
            ).mean(axis=(1, 3))
        return field

    @staticmethod
    def shape(points, contours, fill_rule):
        distance_sq = np.full(points.shape[:2], np.inf, dtype=np.float32)
        winding = np.zeros(points.shape[:2], dtype=np.int32)
        for contour in contours:
            distance_sq = np.minimum(distance_sq, _contour_distance_sq(points, contour))
            winding += _contour_winding(points, contour)
        shape_distance = np.sqrt(distance_sq).astype(np.float32)
        inside = winding != 0 if fill_rule == _NONZERO_FILL else (winding & 1) != 0
        shape_distance[inside] *= -1.0
        return shape_distance


def register_plugins(registry):
    registry.register(
        Plugin(
            "svg",
            PluginFamily.PRIMITIVE,
            TextureKind.SDF,
            params=("source", "center", "height", "samples", "inflate", "oversample"),
            defaults={
                "center": (0.0, 0.0),
                "height": 64.0,
                "samples": 16,
                "inflate": 0.0,
                "oversample": 2,
            },
            render_func=render_svg,
            public=True,
            method_of=(TextureKind.SDF,),
        )
    )
