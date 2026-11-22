"""Rasterize sampled pen strokes alongside ordinary color graph layers."""

from pathlib import Path

from PIL import Image, ImageDraw

from ..gpu.nodes.common import build
from ..util import hex_col

_SUPERSAMPLE = 3


def _stroke_layer(size, viewport, paths):
    width, height = size
    samples = (width * _SUPERSAMPLE, height * _SUPERSAMPLE)
    mask = Image.new("L", samples)
    if not paths:
        return Image.new("RGBA", size)
    pixels_per_x = samples[0] / viewport[0]
    pixels_per_y = samples[1] / viewport[1]
    brush = ImageDraw.Draw(mask)
    for path in paths:
        points = [(x * pixels_per_x, y * pixels_per_y) for x, y in path.points]
        stroke = max(1, round(path.width * (pixels_per_x + pixels_per_y) / 2))
        brush.line(points, width=stroke, fill=255, joint="curve")
        radius = stroke / 2
        for x, y in (points[0], points[-1]):
            brush.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    rgba = tuple(round(channel * 255) for channel in hex_col(paths[0].color))
    color = Image.new("RGBA", size, rgba)
    coverage = mask.resize(size, Image.Resampling.LANCZOS)
    if rgba[3] != 255:
        coverage = coverage.point(lambda value: value * rgba[3] / 255)
    color.putalpha(coverage)
    return color


def _color_layer(renderer, texture):
    with renderer.render(texture.to_rgb()) as result:
        return Image.fromarray(result.read()[::-1].copy())


def _path_layer(layer, renderer, canvas):
    rendered = renderer.render(layer.shape)
    try:
        paths = tuple(
            layer._paths(
                rendered.read(), renderer, canvas.logical_size, canvas.device_scale
            )
        )
    finally:
        if rendered is not layer.shape:
            rendered.close()
    return _stroke_layer(canvas.size, canvas.logical_size, paths)


def save_path_scene(scene, path, canvas, params, progress) -> Path:
    with canvas.session(params=params, progress=progress) as renderer:
        result = _color_layer(renderer, build("clear", scene.background))
        for layer in scene.layers:
            ink = (
                _path_layer(layer, renderer, canvas)
                if layer.hatch and layer.hatch._uses_paths
                else _color_layer(renderer, layer._raster())
            )
            result.alpha_composite(ink)
    destination = Path(path)
    result.save(destination)
    return destination
