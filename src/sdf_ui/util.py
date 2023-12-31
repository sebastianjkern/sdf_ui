__docformat__ = "google"

import numpy as np

def rgb_col(r: int, g: int, b: int, *a):
    """
    Convert RGB values to a normalized tuple.

    Args:
    - r (int): Red value.
    - g (int): Green value.
    - b (int): Blue value.
    - *a: Optional alpha value(s).

    Returns:
    tuple: A tuple of normalized RGB values.

    Example:
    >>> color = rgb_col(255, 0, 0, 128)
    >>> print(color)
    (1.0, 0.0, 0.0, 0.5)
    """
    values = np.array([r, g, b, *a], dtype=float)
    values /= 255
    return tuple(values)


def hex_col(string: str, alpha=255):
    """
    Convert a hexadecimal color string to a normalized tuple.

    Args:
    - string (str): Hexadecimal color string (e.g., "#RRGGBB").
    - alpha (int): Optional alpha value.

    Returns:
    tuple: A tuple of normalized RGB values.

    Example:
    >>> color = hex_col("#FF0000", 128)
    >>> print(color)
    (1.0, 0.0, 0.0, 0.5)
    """
    g = string.lstrip("#")
    col = tuple(int(g[i:i + 2], 16) for i in (0, 2, 4))
    return rgb_col(*col, alpha)


def collinear(x1, y1, x2, y2, x3, y3):
    """
    Check if three points are collinear.

    Args:
    - x1, y1: Coordinates of the first point.
    - x2, y2: Coordinates of the second point.
    - x3, y3: Coordinates of the third point.

    Returns:
    bool: True if collinear, False otherwise.

    Example:
    >>> result = collinear(0, 0, 1, 1, 2, 2)
    >>> print(result)
    True
    """
    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    return area <= 0.000000001
