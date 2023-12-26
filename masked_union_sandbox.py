# %%
import logging

import sys

import random

from functools import reduce

from framework import disc, Context, logger

logger().setLevel(logging.INFO)

SIZE = (512, 512)

# %%

[(random.randint(0, 512), random.randint(0, 512)) for _ in range(10)]

# %%

with Context(SIZE) as context:
    points = [(245, 301),
        (362, 485),
        (306, 292),
        (66, 342),
        (213, 198),
        (303, 18),
        (344, 174),
        (443, 482),
        (163, 294),
        (161, 453)]
    discs = [disc(point, 50) for point in points]

    combined_discs = reduce(lambda x, y: x.union(y), discs)
    combined_discs.save("voronoi.png")
    combined_discs.show()

    objs = list(map(lambda disc: discs[0].masked_union(disc), discs[1:]))
    masks = list(*list(zip(*objs))[1:])
    # list(map(lambda z: z[1].save(f"mask_{z[0]}.png"), enumerate(masks)))

    mask = reduce(lambda m1, m2: m1.multiply(m2), masks)
    mask.save("voronoi_mask.png")
    mask.show()

# %%