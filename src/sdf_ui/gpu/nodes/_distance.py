"""Numerical distance-field corrections for host-rendered SDF nodes."""

import numpy as np


def normalize_distance(field, epsilon, band):
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    band = float(band)
    if not np.isfinite(band) or band <= 0:
        raise ValueError("band must be positive and finite")
    if min(field.shape) < 2:
        raise ValueError("distance normalization needs at least a 2x2 field")
    grad_y, grad_x = np.gradient(field)
    gradient_length = np.hypot(grad_x, grad_y)
    inverse_gradient = np.where(
        gradient_length > epsilon, 1.0 / np.maximum(gradient_length, epsilon), 1.0
    )
    proximity = np.clip(1.0 - np.abs(field) / band, 0.0, 1.0)
    weight = proximity * proximity * (3.0 - 2.0 * proximity)
    scale = 1.0 + weight * (inverse_gradient - 1.0)
    return field * scale


def _add_seeds(seed_x, seed_y, best_distance_sq, mask, px, py, distance_sq):
    better = mask & (distance_sq < best_distance_sq)
    best_distance_sq[better] = distance_sq[better]
    seed_x[better] = px[better]
    seed_y[better] = py[better]


def _jump_pass(seed_x, seed_y, step):
    height, width = seed_x.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    next_x, next_y = seed_x.copy(), seed_y.copy()
    best = np.where(seed_x >= 0, (xx - seed_x) ** 2 + (yy - seed_y) ** 2, np.inf)

    for dy in (-step, 0, step):
        for dx in (-step, 0, step):
            candidate_x = np.roll(seed_x, shift=(dy, dx), axis=(0, 1))
            candidate_y = np.roll(seed_y, shift=(dy, dx), axis=(0, 1))
            valid = candidate_x >= 0
            if dy > 0:
                valid[:dy, :] = False
            elif dy < 0:
                valid[dy:, :] = False
            if dx > 0:
                valid[:, :dx] = False
            elif dx < 0:
                valid[:, dx:] = False
            distance_sq = (xx - candidate_x) ** 2 + (yy - candidate_y) ** 2
            better = valid & (distance_sq < best)
            best[better] = distance_sq[better]
            next_x[better] = candidate_x[better]
            next_y[better] = candidate_y[better]
    return next_x, next_y


def _empty_seeds(field, xx, yy):
    shape = field.shape
    seed_x = np.full(shape, -1.0, dtype=np.float32)
    seed_y = np.full(shape, -1.0, dtype=np.float32)
    best = np.full(shape, np.inf, dtype=np.float32)
    zero = field == 0
    _add_seeds(seed_x, seed_y, best, zero, xx, yy, np.zeros(shape, dtype=np.float32))
    return seed_x, seed_y, best


def _seed_horizontal_edges(field, inside, xx, yy, seeds):
    seed_x, seed_y, best = seeds
    left, right = field[:, :-1], field[:, 1:]
    crossing = inside[:, :-1] != inside[:, 1:]
    fraction = np.abs(left) / np.maximum(np.abs(left) + np.abs(right), 1e-20)
    x, y = xx[:, :-1] + fraction, yy[:, :-1]
    _add_seeds(seed_x[:, :-1], seed_y[:, :-1], best[:, :-1], crossing, x, y, fraction**2)
    _add_seeds(seed_x[:, 1:], seed_y[:, 1:], best[:, 1:], crossing, x, y, (1.0 - fraction) ** 2)


def _seed_vertical_edges(field, inside, xx, yy, seeds):
    seed_x, seed_y, best = seeds
    top, bottom = field[:-1, :], field[1:, :]
    crossing = inside[:-1, :] != inside[1:, :]
    fraction = np.abs(top) / np.maximum(np.abs(top) + np.abs(bottom), 1e-20)
    x, y = xx[:-1, :], yy[:-1, :] + fraction
    _add_seeds(seed_x[:-1, :], seed_y[:-1, :], best[:-1, :], crossing, x, y, fraction**2)
    _add_seeds(seed_x[1:, :], seed_y[1:, :], best[1:, :], crossing, x, y, (1.0 - fraction) ** 2)


def _propagate_seeds(seed_x, seed_y):
    height, width = seed_x.shape
    step = 1 << (max(height, width).bit_length() - 1)
    while step >= 1:
        seed_x, seed_y = _jump_pass(seed_x, seed_y, step)
        step >>= 1
    return _jump_pass(seed_x, seed_y, 1)


def redistance(field):
    """Approximate signed distance to a sampled field's subpixel zero contour."""
    height, width = field.shape
    if min(field.shape) < 2:
        raise ValueError("redistance needs at least a 2x2 field")
    inside = field < 0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    seeds = _empty_seeds(field, xx, yy)
    _seed_horizontal_edges(field, inside, xx, yy, seeds)
    _seed_vertical_edges(field, inside, xx, yy, seeds)
    seed_x, seed_y, _best = seeds
    if not np.any(seed_x >= 0):
        raise ValueError("redistance requires a zero contour in the sampled field")
    seed_x, seed_y = _propagate_seeds(seed_x, seed_y)
    distance = np.hypot(xx - seed_x, yy - seed_y)
    return np.where(inside, -distance, distance).astype(np.float32)
