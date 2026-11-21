"""Rotation math shared by grid and rectangle shader extensions."""


def rotate(x, y, angle):
    """Apply the row-vector rotation used by the native shader."""
    return x * angle.cos() - y * angle.sin(), x * angle.sin() + y * angle.cos()


