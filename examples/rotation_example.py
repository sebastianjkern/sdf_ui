import math

from src.sdf_ui import Context, rounded_rect

def rotation_example():
    SIZE = (512, 512)

    with Context(SIZE) as ctx:
        sdf = rounded_rect(ctx, (ctx.percent_of_min(50), ctx.percent_of_min(50)), (50, 50), (ctx.percent_of_min(1), ctx.percent_of_min(1), ctx.percent_of_min(1),
                            ctx.percent_of_min(1)), 25/360*math.pi)
        sdf.fill((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0), 10).show()