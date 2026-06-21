# %% 
from functools import reduce

import random

from sdf_ui import Canvas, sdf

def voronoi_example():
    with Canvas((512, 512)) as ctx:
        points = [(random.randint(0, 512), random.randint(0, 512)) for _ in range(100)]
        discs = [sdf.circle(point, 20).cache(f"disc_{index}") for index, point in enumerate(points)]

        combined_discs = reduce(lambda x, y: x | y, discs).cache("combined_discs")
        filled = combined_discs.fill((1, 1, 1, 1), (0.1, 0.1, 0.1, 1.0), inflate=25, inner=0, outer=50)
        cache = {}
        combined_discs.show(ctx, cache=cache)
        filled.show(ctx, cache=cache)
