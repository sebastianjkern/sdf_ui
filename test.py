from src.sdf_ui import *

from random import random

import logging

SIZE = (1920, 1080)

logger().setLevel(logging.CRITICAL)

def striped_circle(context, color=(1, 0, 0, 1), center=(SIZE[0]/2, SIZE[1]/2), radius=50):
    transparent = (*color[:3], 0)

    line_thickness = 2
    gap = 5

    sdf = disc(context, center, radius)
    
    background = clear_color(context, (0, 0, 0, 0))

    for i in range(int(radius/(line_thickness + gap))):
        outline = sdf.outline(color, transparent, inflate=-i*(line_thickness+gap))
        background = background.alpha_overlay(outline)

    return background


if __name__ == "__main__":
    with Context(SIZE) as ctx:
        bg = clear_color(ctx, (1, 1, 1, 1))

        for _ in range(10):
            color = (random(), random(), random(), 1)
            center = (random() * SIZE[0], random()* SIZE[1])
            circle = striped_circle(ctx, color=color, center=center, radius=ctx.percent(10))
            bg = bg.alpha_overlay(circle)

        bg.show()
        