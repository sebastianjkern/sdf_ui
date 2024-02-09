from src.sdf_ui import *

from random import random

import logging

SIZE = (500, 500)

logger().setLevel(logging.DEBUG)


with Context(SIZE) as ctx:
    sdf = disc(ctx, (ctx.pt * 0.5, ctx.pt * 0.5), ctx.pt * 0.5)
    repeated = sdf.repeat(s=ctx.pt*2)
    filled = repeated.fill((1, 1, 1, 1), (0, 0, 0, 1), 0)
    filled.show()
    filled.save("repetition.png")
