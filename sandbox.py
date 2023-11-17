import logging
import random
import sys
import time
import math

from framework import Context, set_context, clear_color, logger, radial_gradient, hex_col, bezier, linear_gradient, \
    grid, rounded_rect, disc, film_grain, glyph_sdf

size = (1920, 1080)

logger().setLevel(logging.INFO)


def rand_color():
    red = random.randint(0, 255) / 255
    green = random.randint(0, 255) / 255
    blue = random.randint(0, 255) / 255
    return red, green, blue


def rand_point():
    x = random.randint(0, size[0])
    y = random.randint(0, size[1])

    return x, y


COLORS = [
    "#62bb47",
    "#fcb827",
    "#f6821f",
    "#e03a3c",
    "#963d97",
    "#009ddc"
]

context = Context(size)
set_context(context)

# Example 1

size = (1920, 1080)

context = Context(size)
set_context(context)

start = time.time_ns()

backdrop = radial_gradient((size[0] / 2, size[1] / 2), hex_col("#004C81", alpha=255),
                           hex_col("#062A4A", alpha=255), inner=0,
                           outer=context.percent_of_min(75)).alpha_overlay(
    grid((10, 10), (50, 50)).fill(hex_col("#5EC6E2", 255), hex_col("#5EC6E2", 0), inflate=.5)) \
    .alpha_overlay(grid((10, 10), (150, 150)).fill(hex_col("#5EC6E2", 255), hex_col("#5EC6E2", 0), inflate=1.5))

gradient = linear_gradient((100, 100), (900, 100), (*rand_color(), 1.0), (*rand_color(), 1.0)).transparency(0.45)

blur = backdrop.blur(n=15, base=13)

mask_sdf = rounded_rect((int(size[0] / 2), int(size[1] / 2)),
                        (int(size[1] / 3), int(size[1] / 3)),
                        (context.percent_of_min(10), context.percent_of_min(10), context.percent_of_min(10),
                         context.percent_of_min(10)))

mask = mask_sdf.generate_mask()
outline = mask_sdf.outline((1.0, 1.0, 1.0, .25), (1.0, 1.0, 1.0, 0.0), inflate=-1.5)

glass_col = (0.75, 0.75, 0.75, 0.75)
glass = mask_sdf.fill(glass_col, (0.0, 0.0, 0.0, 0.0), 0)

glass_shadow = mask_sdf.shadow(25, 0, 0.75)

mask_sdf.fill_from_texture(gradient)

for index in range(10):
    for i in range(index + 1):
        x, y = random.randint(50, size[0] - 50), random.randint(50, size[1] - 50)
        r = random.randint(50, 100)

        col0 = hex_col(random.choice(COLORS))[:3]
        col1 = (*col0, 1.0)
        col2 = (*col0, 0.0)

        distance = disc((x, y), r)
        shade = distance.shadow(7, 0, transparency=0.5)

        backdrop1 = (backdrop if i == 0 else backdrop1).alpha_overlay(shade).alpha_overlay(distance.fill(col1, col2, 0))

    image = backdrop1.alpha_overlay(glass_shadow).mask(blur, mask).alpha_overlay(glass).alpha_overlay(
        outline).alpha_overlay(film_grain().to_lab().transparency(5 / 255)).to_rgb()

    image.save(f"{str(index).zfill(3)}_img.png")

    end = time.time_ns()
    print((end - start) / 1e9)
    start = end

# Example 2
context = Context((1080, 1080))
set_context(context)

scale = 0.65
offset_x = 25
offset_y = 25

inflation = 7.5

sdf = glyph_sdf("i", scale, offset_x, offset_y, path="fonts/SFUIDisplay-Bold.ttf")

sdf.save()

bg = clear_color(hex_col("#2C2D35"))
shadow = sdf.shadow(distance=10, inflate=inflation, transparency=1.0)
colored_glyph = sdf.fill(hex_col("#e9c46a"), (0.0, 0.0, 0.0, 0.0), inflation)

bg.alpha_overlay(shadow).alpha_overlay(colored_glyph).to_rgb().show()

sys.exit()

# Example 3

# Animation example
#

context = Context((int(1280 / 2), int(720 / 2)))
set_context(context)

background = clear_color((1.0, 1.0, 1.0, 1.0))

disc1 = disc((percent_x(10), percent_y(50)), percent_of_min(35))

images = []

for i in range(1000):
    name_of_frame = f"out/anim/{str(i).zfill(5)}.png"

    disc2 = disc(
        (context.percent_x(35) + context.percent_x(25 * math.sin(i / (math.pi * 2 * 15))), context.percent_y(50)),
        context.percent_of_min(15))

    col = hex_col("#62bb47")[:3]
    col1 = (*col, 1.0)
    col2 = (*col, 0.0)

    image = background.alpha_overlay(disc1.smooth_union(disc2).fill(col1, col2))
    image.to_rgb().save(name_of_frame)
    images.append(name_of_frame)

convert_to_video("out", images)
