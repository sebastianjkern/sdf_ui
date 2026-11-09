"""Analytic autodiff rules for standalone primitive compute shaders.

``RULES[name](context, inputs, params)`` uses registry shader names and raw
GLSL uniform names. Context exposes Tensor pixel grids ``x``, ``y`` and
``size=(width, height)``; input images are scalar ``(height, width)`` Tensors.
The masked union returns ``(distance, rgba_mask)`` before rgba8 quantization.

Discrete angle masks, integer addresses and closest-point selection are frozen
for each evaluation. Gradients are exact within each smooth branch. Norms and
absolute values use zero at the origin; hard extrema split ties equally among
their active candidates. Repeat spacing and transform kind have zero derivatives.
Bézier roots are numerical branch decisions, with analytic envelope gradients
through the selected curve points, rather than numerical differentiation.
"""

import numpy as np

from .autodiff import Tensor, maximum, minimum, stack, where
from .render import _clip, _length, _mix, _smoothstep, _tensor

__all__ = ["RULES"]

_TAU = 6.28318530718


def _xy(context, point):
    point = _tensor(point)
    return context.x - point[0], context.y - point[1]


def _rotate(x, y, angle):
    """The row-vector rotation used by rect/grid (GLSL column-major mat2)."""
    return x * angle.cos() - y * angle.sin(), x * angle.sin() + y * angle.cos()


def _minimum_candidates(*values):
    """Symmetric active-set derivative, including more than two equal minima."""
    values = stack(values, axis=-1)
    data = values.data
    active = data == data.min(axis=-1, keepdims=True)
    return where(active, values, 0.0).sum(axis=-1) / active.sum(axis=-1)


def _segment(x, y, ax, ay, bx, by, epsilon=None):
    px, py = x - ax, y - ay
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if epsilon is not None:
        denominator = maximum(denominator, epsilon)
    elif np.any(denominator.data == 0):
        raise ValueError("line/triangle requires distinct segment endpoints")
    t = _clip((px * dx + py * dy) / denominator, 0.0, 1.0)
    return _length(px - t * dx, py - t * dy)


def _circle(context, inputs, params):
    return _length(*_xy(context, params["offset"])) - _tensor(params["radius"])


def _rect(context, inputs, params):
    x, y = _rotate(*_xy(context, params["offset"]), _tensor(params["angle"]))
    size, radius = _tensor(params["size"]), _tensor(params["corner_radius"])
    # Accept the scalar splat as well as the actual vec4 uniform.
    if radius.shape == (4,):
        hx, hy = _smoothstep(0.0, 0.0001, x), _smoothstep(0.0, 0.0001, y)
        radius = _mix(
            _mix(radius[0], radius[2], hx), _mix(radius[1], radius[3], hx), hy
        )
    elif radius.shape != ():
        raise ValueError("corner_radius must be a scalar or vec4")
    qx, qy = x.abs() - size[0] + radius, y.abs() - size[1] + radius
    return (
        minimum(maximum(qx, qy), 0.0)
        + _length(maximum(qx, 0.0), maximum(qy, 0.0))
        - radius
    )


def _ellipse(context, inputs, params):
    x, y = _xy(context, params["center"])
    radii = maximum(_tensor(params["radii"]).abs(), 1e-6)
    f = (x / radii[0]) ** 2 + (y / radii[1]) ** 2 - 1.0
    return f / maximum(_length(2.0 * x / radii[0] ** 2, 2.0 * y / radii[1] ** 2), 1e-6)


def _line(context, inputs, params):
    a, b = _tensor(params["a"]), _tensor(params["b"])
    return _segment(context.x, context.y, a[0], a[1], b[0], b[1])


def _capsule(context, inputs, params):
    a, b = _tensor(params["a"]), _tensor(params["b"])
    return _segment(context.x, context.y, a[0], a[1], b[0], b[1], 1e-6) - _tensor(
        params["radius"]
    )


def _half_plane(context, inputs, params):
    x, y = _xy(context, params["point"])
    normal = _tensor(params["normal"])
    norm = maximum(_length(normal[0], normal[1]), 1e-6)
    return (x * normal[0] + y * normal[1]) / norm


def _ring(context, inputs, params):
    return (
        _length(*_xy(context, params["center"])) - _tensor(params["radius"])
    ).abs() - _tensor(params["thickness"]) * 0.5


def _angles(x, y, params):
    start, end = _tensor(params["start_angle"]), _tensor(params["end_angle"])
    angle = np.mod(np.arctan2(y.data, x.data), _TAU)
    lo, hi = np.mod(start.data, _TAU), np.mod(end.data, _TAU)
    inside = np.where(
        lo <= hi, (angle >= lo) & (angle <= hi), (angle >= lo) | (angle <= hi)
    )
    # sin/cos are periodic; reducing their arguments only changes branch masks.
    return start, end, inside


def _arc(context, inputs, params):
    x, y = _xy(context, params["center"])
    start, end, inside = _angles(x, y, params)
    radius = _tensor(params["radius"])
    endpoints = minimum(
        _length(x - radius * start.cos(), y - radius * start.sin()),
        _length(x - radius * end.cos(), y - radius * end.sin()),
    )
    return where(inside, (_length(x, y) - radius).abs(), endpoints)


def _sector(context, inputs, params):
    x, y = _xy(context, params["center"])
    start, end, inside = _angles(x, y, params)
    radius = _tensor(params["radius"])
    radial = _length(x, y)
    distance = _minimum_candidates(
        (radial - radius).abs(),
        _segment(x, y, 0.0, 0.0, radius * start.cos(), radius * start.sin(), 1e-6),
        _segment(x, y, 0.0, 0.0, radius * end.cos(), radius * end.sin(), 1e-6),
    )
    return where(inside & (radial.data <= radius.data), -distance, distance)


def _grid(context, inputs, params):
    x, y = _rotate(context.x, context.y, _tensor(params["angle"]))
    size, offset = _tensor(params["grid_size"]), _tensor(params["offset"])
    if np.any(size.data == 0):
        raise ValueError("grid_size must be non-zero")
    x, y = x - offset[0], y - offset[1]
    # GLSL mod is floor-based, including for negative coordinates/spacings.
    x = x - size[0] * np.floor((x / size[0]).data) - size[0] * 0.5
    y = y - size[1] * np.floor((y / size[1]).data) - size[1] * 0.5
    return minimum(x.abs(), y.abs())


def _triangle(context, inputs, params):
    points = [_tensor(params[name]) for name in ("point0", "point1", "point2")]
    e0, e2 = points[1] - points[0], points[0] - points[2]
    orientation = np.sign((e0[0] * e2[1] - e0[1] * e2[0]).data)
    if orientation == 0:
        raise ValueError("triangle requires non-collinear vertices for differentiation")
    distances, sides = [], []
    for a, b in zip(points, points[1:] + points[:1]):
        distances.append(_segment(context.x, context.y, a[0], a[1], b[0], b[1]))
        x, y = _xy(context, a)
        edge = b - a
        sides.append(orientation * (x * edge[1] - y * edge[0]).data)
    return -_minimum_candidates(*distances) * np.sign(np.minimum.reduce(sides))


def _bezier(context, inputs, params):
    a, b, c = (_tensor(params[name]) for name in ("a", "b", "c"))
    # Match the shader's componentwise perturbation for zero quadratic terms.
    b = b + np.where((2.0 * b - a - c).data == 0.0, 1e-4, 0.0)
    av, bv, cv = a.data, b.data, c.data
    chord, control = cv - av, bv - av
    if chord[0] * control[1] - chord[1] * control[0] == 0:
        raise ValueError("bezier requires non-collinear adjusted control points")
    linear, quadratic = 2.0 * (bv - av), av - 2.0 * bv + cv
    delta = np.stack((context.x.data - av[0], context.y.data - av[1]), axis=-1)
    # (B(t)-p).B'(t)=0 is cubic. Roots determine only the active point;
    # the envelope theorem removes dt/dtheta at each stationary minimum.
    leading = 2.0 * np.dot(quadratic, quadratic)
    second = 3.0 * np.dot(linear, quadratic)
    third = np.dot(linear, linear) - 2.0 * (delta @ quadratic)
    fourth = -(delta @ linear)
    candidates = np.empty(context.x.shape + (3,))
    for index in np.ndindex(context.x.shape):
        roots = np.roots((leading, second, third[index], fourth[index]))
        real = np.sort(roots.real[np.abs(roots.imag) <= 1e-9])
        # One real root is repeated by solveCubic in the GLSL branch.
        candidates[index] = np.clip(np.resize(real, 3), 0.0, 1.0)
    # Clamped duplicate roots represent one candidate, not extra tie weight.
    candidates.sort(axis=-1)
    distances = []
    for i in range(3):
        t = candidates[..., i]
        q = a * (1.0 - t[..., None]) ** 2 + 2.0 * b * (t * (1.0 - t))[..., None]
        q = q + c * (t * t)[..., None]
        distances.append(_length(context.x - q[..., 0], context.y - q[..., 1]))
    values = stack(distances, axis=-1)
    unique = np.concatenate(
        (np.ones(context.x.shape + (1,), dtype=bool), np.diff(candidates) != 0.0),
        axis=-1,
    )
    data = values.data
    active = unique & (data == data.min(axis=-1, keepdims=True))
    return where(active, values, 0.0).sum(axis=-1) / active.sum(axis=-1)


def _abs(context, inputs, params):
    return inputs["sdf0"].abs()


def _union(context, inputs, params):
    return minimum(inputs["sdf0"], inputs["sdf1"])


def _intersection(context, inputs, params):
    return maximum(inputs["sdf0"], inputs["sdf1"])


def _subtract(context, inputs, params):
    return maximum(inputs["sdf0"], -inputs["sdf1"])


def _interpolate(context, inputs, params):
    return _mix(inputs["sdf0"], inputs["sdf1"], _tensor(params["t"]))


def _smooth_min(context, inputs, params):
    first, second = inputs["sdf0"], inputs["sdf1"]
    k = maximum(_tensor(params["smoothness"]), 1e-6)
    h = _clip(0.5 + 0.5 * (second - first) / k, 0.0, 1.0)
    return _mix(second, first, h) - k * h * (1.0 - h)


def _masked_union(context, inputs, params):
    distance = _union(context, inputs, params)
    mask = _smoothstep(0.0, 1.0, inputs["sdf1"] - distance)
    return distance, stack((mask, mask, mask, mask * 0.0 + 1.0), axis=-1)


def _load_zero(source, x, y):
    """imageLoad uses truncation toward zero and zero for out-of-bounds reads."""
    ix, iy = np.trunc(x).astype(np.int64), np.trunc(y).astype(np.int64)
    height, width = source.shape
    valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    return where(
        valid, source[np.clip(iy, 0, height - 1), np.clip(ix, 0, width - 1)], 0.0
    )


def _repeat(context, inputs, params):
    spacing = _tensor(params["repeat"])
    if spacing.item() == 0.0:
        raise ValueError("repeat spacing must be non-zero")
    x, y, s = context.x.data, context.y.data, spacing.item()
    # GLSL round permits either direction at exact halves; use nearest-even.
    ix, iy = np.rint(x / s), np.rint(y / s)
    ox, oy = np.sign(x - s * ix), np.sign(y - s * iy)
    candidates = [Tensor(np.full(context.x.shape, 1e20))]
    for j in range(2):
        for i in range(2):
            candidates.append(
                _load_zero(inputs["sdf0"], x - s * (ix + i * ox), y - s * (iy + j * oy))
            )
    return _minimum_candidates(*candidates) + spacing * 0.0


def _bilinear(source, x, y):
    height, width = source.shape
    x, y = _clip(x, 0.0, width - 1), _clip(y, 0.0, height - 1)
    ix, iy = np.floor(x.data).astype(np.int64), np.floor(y.data).astype(np.int64)
    jx, jy = np.minimum(ix + 1, width - 1), np.minimum(iy + 1, height - 1)
    tx, ty = x - ix, y - iy
    return _mix(
        _mix(source[iy, ix], source[iy, jx], tx),
        _mix(source[jy, ix], source[jy, jx], tx),
        ty,
    )


def _transform(context, inputs, params):
    kind = _tensor(params["transform_kind"])
    mode, scale = kind.item(), 1.0
    if mode < 0.5:
        x, y = _xy(context, params.get("offset", (0.0, 0.0)))
    else:
        center = _tensor(params.get("center", (0.0, 0.0)))
        x, y = _xy(context, center)
        if mode < 1.5:
            factor = _tensor(params.get("factor", 0.0))
            if factor.item() == 0.0:
                raise ValueError("scale factor must be non-zero")
            x, y, scale = x / factor, y / factor, factor.abs()
        elif mode < 2.5:
            angle = _tensor(params.get("angle", 0.0))
            x, y = angle.cos() * x + angle.sin() * y, -angle.sin() * x + angle.cos() * y
        else:
            skew = _tensor(params.get("skew", (0.0, 0.0)))
            determinant = 1.0 - skew[0] * skew[1]
            if determinant.item() == 0.0:
                raise ValueError("skew values produce a singular transform")
            x, y = (x - skew[0] * y) / determinant, (y - skew[1] * x) / determinant
            # At exactly 2.5 inverse_transform skews but distance_scale is 1.
            if mode > 2.5:
                scale = maximum(
                    0.5
                    * (
                        _length(1.0 / determinant, -skew[1] / determinant)
                        + _length(-skew[0] / determinant, 1.0 / determinant)
                    ),
                    1e-6,
                )
        x, y = x + center[0], y + center[1]
    return _bilinear(inputs["sdf"], x, y) * scale + kind * 0.0


RULES = {
    "abs": _abs,
    "arc": _arc,
    "bezier": _bezier,
    "capsule": _capsule,
    "circle": _circle,
    "ellipse": _ellipse,
    "grid": _grid,
    "half_plane": _half_plane,
    "interpolation": _interpolate,
    "intersection": _intersection,
    "line": _line,
    "masked_union": _masked_union,
    "rect": _rect,
    "repeat": _repeat,
    "ring": _ring,
    "sector": _sector,
    "smooth_min": _smooth_min,
    "subtract": _subtract,
    "transform": _transform,
    "triangle": _triangle,
    "union": _union,
}
