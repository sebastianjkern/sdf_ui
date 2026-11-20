"""A progress indicator with a fixed track and a parameterized fill."""

from numbers import Real

from ..core.color import ColorTexture
from ..core.expressions import (
    _lazy_scalar,
    _scalar_maximum,
    _scalar_minimum,
    _select_positive,
)
from .panel import Panel


def _cache_when_enabled(texture, cache):
    return texture if cache else texture.uncached()


def _progress_track(box, radius, fill_color, track_color, cache):
    x, y, width, height = box
    shape = Panel((x, y, width, height), radius=radius).shape()
    shape = _cache_when_enabled(shape, cache)
    return _cache_when_enabled(shape.fill(track_color, (0.0, 0.0, 0.0, 0.0)), cache), shape


def _progress_fill(box, fraction, radius, fill_color, track_shape, cache):
    x, y, width, height = box
    shape = Panel((x, y, width * fraction, height), radius=radius).shape()
    shape = _cache_when_enabled(shape, cache).intersection(track_shape)
    shape = _cache_when_enabled(shape, cache)
    fill = _cache_when_enabled(shape.fill(fill_color, (0.0, 0.0, 0.0, 0.0)), cache)
    if not isinstance(fraction, Real):
        fill = fill.transparency(_select_positive(fraction, 1.0, 0.0))
    return fill


def progress_bar(
    box,
    value,
    *,
    fill_color="#62dfcf",
    track_color="#364356",
    radius=None,
    cache: bool = True,
) -> ColorTexture:
    """Draw progress inside a full canvas box, clamped to [0, 1]."""
    x, y, width, height = map(_lazy_scalar, box)
    normalized_box = (x, y, width, height)
    fraction = _scalar_minimum(_scalar_maximum(_lazy_scalar(value), 0.0), 1.0)
    radius = height / 2 if radius is None else radius
    track, track_shape = _progress_track(
        normalized_box, radius, fill_color, track_color, cache
    )
    if isinstance(fraction, Real) and fraction == 0:
        return track
    fill = _progress_fill(normalized_box, fraction, radius, fill_color, track_shape, cache)
    composite = track.alpha_overlay(_cache_when_enabled(fill, cache))
    return _cache_when_enabled(composite, cache)
