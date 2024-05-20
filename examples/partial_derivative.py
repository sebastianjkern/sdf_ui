# %% 
from functools import reduce

import random

from src.sdf_ui import Context, disc

def partial_derivative_example():
    with Context((1080, 1080)) as ctx:
        points = [(random.randint(0, 1080), random.randint(0, 1080)) for _ in range(100)]
        discs = [disc(ctx, point, 20) for point in points]

        combined_discs = reduce(lambda x, y: x | y, discs)
        combined_discs.partial_derivative().show()