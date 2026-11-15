"""Shared validation and velocity evaluation for vortex field operations."""

import numpy as np


def normalize_vortices(value):
    vortices = np.asarray(value, dtype=np.float32)
    if vortices.ndim != 2 or vortices.shape[1] != 3 or not len(vortices):
        raise ValueError("vortices must be a non-empty sequence of (x, y, strength)")
    if not np.isfinite(vortices).all():
        raise ValueError("vortex positions and strengths must be finite")
    return vortices


def validate_radius(value):
    radius = float(value)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("vortex radius must be positive and finite")
    return radius


def vortex_velocity(x, y, vortices, radius):
    vx = np.zeros_like(x, dtype=np.float32)
    vy = np.zeros_like(y, dtype=np.float32)
    for cx, cy, strength in vortices:
        dx, dy = x - cx, y - cy
        influence = np.exp(-(dx * dx + dy * dy) / (2.0 * radius * radius))
        vx += -strength * dy / radius * influence
        vy += strength * dx / radius * influence
    return vx, vy
