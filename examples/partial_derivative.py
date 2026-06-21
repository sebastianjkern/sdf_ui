# %% 
from functools import reduce

import random

from sdf_ui import Canvas, sdf

def partial_derivative_example():
    with Canvas((1080, 1080)) as ctx:
        points = [(random.randint(0, 1080), random.randint(0, 1080)) for _ in range(100)]
        discs = [sdf.circle(point, 20).cache(f"disc_{index}") for index, point in enumerate(points)]

        combined_discs = reduce(lambda x, y: x | y, discs).cache("combined_discs")
        combined_discs.partial_derivative().show(ctx)
