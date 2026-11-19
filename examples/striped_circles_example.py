from random import random

from sdf_ui import Canvas, color

SIZE = (1920, 1080)


def striped_circles_example():
    with Canvas(SIZE) as ctx:
        image = color.clear((1, 1, 1, 1)).cache("background")

        for _ in range(10):
            circle_color = (random(), random(), random(), 1)
            center = (random() * SIZE[0], random() * SIZE[1])
            radius = ctx.percent(10)
            circle = color.concentric_rings(
                center,
                radius,
                count=int(radius / 7),
                tint=circle_color,
                line_width=2,
                gap=5,
            )
            image = image.alpha_overlay(circle)

        image.show(canvas=ctx)
