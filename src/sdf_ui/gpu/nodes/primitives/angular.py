"""Angle selection shared by arc and sector shader extensions."""


def angles(x, y, params):
    import numpy as np

    from sdf_ui.learning.render import _tensor

    start, end = _tensor(params["start_angle"]), _tensor(params["end_angle"])
    tau = 6.28318530718
    angle = np.mod(np.arctan2(y.data, x.data), tau)
    lo, hi = np.mod(start.data, tau), np.mod(end.data, tau)
    inside = np.where(
        lo <= hi, (angle >= lo) & (angle <= hi), (angle >= lo) | (angle <= hi)
    )
    return start, end, inside


