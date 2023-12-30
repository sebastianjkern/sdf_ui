# %% 
from functools import reduce

import random
import logging

from framework import Context, disc, logger

logger().setLevel(logging.CRITICAL)

with Context((512, 512)) as context:
    points = [(random.randint(0, 512), random.randint(0, 512)) for _ in range(100)]
    discs = [disc(point, 20) for point in points]

    combined_discs = reduce(lambda x, y: x | y, discs)
    combined_discs.save("voronoi.png")

    filled = combined_discs.fill((1, 1, 1, 1), (0.1, 0.1, 0.1, 1.0), inflate=25, inner=0, outer=50)
    filled.to_rgb().save("voronoi_filled.png")
