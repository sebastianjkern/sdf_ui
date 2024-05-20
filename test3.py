from src.sdf_ui import *

from random import random

import logging

SIZE = (512, 512)

logger().setLevel(logging.INFO)


with Context(SIZE) as ctx:
    shapes = glyph_mask(ctx, "5", 0.25, 10, 10)
    background = clear_color(ctx, (0, 0, 0, 1))
    foreground = clear_color(ctx, (1, 1, 1, 1))

    seven = glyph_sdf(ctx, "7", 0.25, 10, 10).fill((1, 1, 1, 1), (0, 0, 0, 1), inflate=1)
    seven.save("7.png")
