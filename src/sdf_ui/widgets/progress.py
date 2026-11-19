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


def progress_bar(
    box,
    value,
    *,
    fill_color="#62dfcf",
    track_color="#364356",
    radius=None,
    cache: bool = True,
) -> ColorTexture:
    """Draw progress inside a full canvas box.

    Clamp ``value`` to [0, 1]. The value and box coordinates accept lazy
    expressions. By default the corner radius is half the track height.
    Use ``cache=False`` to avoid accumulating cached layers in an animation.
    """
    x, y, width, height = map(_lazy_scalar, box)
    fraction = _scalar_minimum(_scalar_maximum(_lazy_scalar(value), 0.0), 1.0)
    radius = height / 2 if radius is None else radius
    track_shape = Panel((x, y, width, height), radius=radius).shape()
    if not cache:
        track_shape = track_shape.uncached()
    transparent = (0.0, 0.0, 0.0, 0.0)
    track = track_shape.fill(track_color, transparent)
    if not cache:
        track = track.uncached()
    if isinstance(fraction, Real) and fraction == 0:
        return track

    fill_shape = Panel((x, y, width * fraction, height), radius=radius).shape()
    if not cache:
        fill_shape = fill_shape.uncached()
    fill_shape = fill_shape.intersection(track_shape)
    if not cache:
        fill_shape = fill_shape.uncached()
    fill = fill_shape.fill(fill_color, transparent)
    if not isinstance(fraction, Real):
        if not cache:
            fill = fill.uncached()
        fill = fill.transparency(_select_positive(fraction, 1.0, 0.0))
    composite = track.alpha_overlay(fill if cache else fill.uncached())
    return composite if cache else composite.uncached()
