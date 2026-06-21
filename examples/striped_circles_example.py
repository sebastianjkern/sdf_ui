from random import random

from sdf_ui import Canvas, color, sdf


SIZE = (1920, 1080)


def striped_circle(circle_color, center, radius):
    transparent = (*circle_color[:3], 0)
    line_thickness = 2
    gap = 5

    circle = sdf.circle(center, radius).cache("circle")
    image = color.clear((0, 0, 0, 0)).cache("stripe_background")

    for i in range(int(radius / (line_thickness + gap))):
        outline = circle.outline(circle_color, transparent, inflate=-i * (line_thickness + gap))
        image = image.alpha_overlay(outline)

    return image


def striped_circles_example():
    with Canvas(SIZE) as ctx:
        image = color.clear((1, 1, 1, 1)).cache("background")

        for _ in range(10):
            circle_color = (random(), random(), random(), 1)
            center = (random() * SIZE[0], random() * SIZE[1])
            circle = striped_circle(circle_color, center, ctx.percent(10))
            image = image.alpha_overlay(circle)

        image.show(ctx)
