"""Reference sampling shared by the stencil-based shading extensions."""


def sample(value, dx, dy, *, zero=False):
    """Integer imageLoad with explicit zero or clamped boundary behavior."""
    import numpy as np

    from sdf_ui.learning.autodiff import where

    height, width = value.shape[:2]
    y, x = np.indices((height, width))
    x, y = x + dx, y + dy
    result = value[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]
    if zero:
        result = where((x >= 0) & (x < width) & (y >= 0) & (y < height), result, 0.0)
    return result
