"""Segment distance shared by line, capsule, and related primitives."""


def minimum_candidates(*values):
    """Split the gradient evenly among equal minimum candidates."""
    from sdf_ui.learning.autodiff import stack, where

    values = stack(values, axis=-1)
    data = values.data
    active = data == data.min(axis=-1, keepdims=True)
    return where(active, values, 0.0).sum(axis=-1) / active.sum(axis=-1)


def segment(x, y, ax, ay, bx, by, epsilon=None):
    import numpy as np

    from sdf_ui.learning.autodiff import maximum
    from sdf_ui.learning.render import _clip, _length

    px, py = x - ax, y - ay
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if epsilon is not None:
        denominator = maximum(denominator, epsilon)
    elif np.any(denominator.data == 0):
        raise ValueError("line/triangle requires distinct segment endpoints")
    t = _clip((px * dx + py * dy) / denominator, 0.0, 1.0)
    return _length(px - t * dx, py - t * dy)

