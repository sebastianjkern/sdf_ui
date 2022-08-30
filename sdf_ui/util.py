import logging

import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s :: %(message)s')


def logger():
    return logging.getLogger(__name__)


class Singleton(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance


def rgb_col(r: int, g: int, b: int, *a):
    values = np.array([r, g, b, *a], dtype=float)
    values /= 255
    return tuple(values)


def hex_col(string: str):
    g = string.lstrip("#")
    col = tuple(int(g[i:i + 2], 16) for i in (0, 2, 4))
    return rgb_col(*col, 255)
