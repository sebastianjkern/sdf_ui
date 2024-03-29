# %% 
from functools import reduce

import random

from src.sdf_ui import Context, disc

def voronoi_example():
    with Context((512, 512)) as ctx:
        points = [(random.randint(0, 512), random.randint(0, 512)) for _ in range(100)]
        discs = [disc(ctx, point, 20) for point in points]

        combined_discs = reduce(lambda x, y: x | y, discs)
        combined_discs.show()

        filled = combined_discs.fill((1, 1, 1, 1), (0.1, 0.1, 0.1, 1.0), inflate=25, inner=0, outer=50)
        filled.show()
