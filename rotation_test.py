import logging
import math

from framework import Context, logger, rounded_rect

logger().setLevel(logging.INFO)

SIZE = (512, 512)

with Context(SIZE) as context:
    sdf = rounded_rect((context.percent_of_min(50), context.percent_of_min(50)), (50, 50), (context.percent_of_min(1), context.percent_of_min(1), context.percent_of_min(1),
                         context.percent_of_min(1)), 25/360*math.pi)
    sdf.fill((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0), 10).to_rgb().show()