"""Shared text primitives for chart examples."""

from sdf_ui import sdf


def chart_text(value, x, y, size, tint="#f4f6fb"):
    return sdf.text(value, ox=x, oy=y, size=size).ink(tint, inflate=0.2)
