import math

from sdf_ui import Canvas, sdf

def rotation_example():
    SIZE = (512, 512)

    with Canvas(SIZE) as ctx:
        shape = sdf.rect(
            (ctx.percent_of_min(50), ctx.percent_of_min(50)),
            (50, 50),
            (
                ctx.percent_of_min(1),
                ctx.percent_of_min(1),
                ctx.percent_of_min(1),
                ctx.percent_of_min(1),
            ),
            25 / 360 * math.pi,
        ).cache("rotated_rect")
        shape.fill((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0), 10).show(context)
