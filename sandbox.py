import logging
import random
import time

from framework.context import Context
from framework.log import logger
from framework.main import set_context, clear_color, radial_gradient, bezier, film_grain, percent_of_min, grid, disc, \
    linear_gradient, rounded_rect
from framework.util import hex_col

size = (1080, 1080)

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
col0 = random.choice(COLORS)
col1 = random.choice(COLORS)
col2 = random.choice(COLORS)
col3 = random.choice(COLORS)

image = clear_color(hex_col(random.choice(COLORS))) \
    .alpha_overlay(
    radial_gradient((100, 100), hex_col(col0, alpha=150), hex_col(col0, alpha=0.0), inner=50, outer=750)) \
    .alpha_overlay(
    radial_gradient((750, 500), hex_col(col1, alpha=255), hex_col(col1, alpha=0.0), inner=50, outer=750)) \
    .alpha_overlay(
    radial_gradient((100, 750), hex_col(col2, alpha=180), hex_col(col2, alpha=0.0), inner=50, outer=750)) \
    .alpha_overlay(
    bezier(rand_point(), rand_point(), rand_point()).fill(hex_col(col3, alpha=150), hex_col(col3, alpha=0),
                                                          inflate=0, inner=0, outer=250)) \
    .alpha_overlay(film_grain().transparency(10 / 255))

image.to_rgb()  # .show()

# Example 2

size = (1920, 1080)

context = Context(size)
set_context(context)

for _ in range(1):
    start = time.time_ns()

    backdrop = radial_gradient((size[0] / 2, size[1] / 2), hex_col("#004C81", alpha=255),
                               hex_col("#062A4A", alpha=255), inner=0,
                               outer=percent_of_min(75)).alpha_overlay(
        grid((10, 10), (50, 50)).fill(hex_col("#5EC6E2", 255), hex_col("#5EC6E2", 0), inflate=.5)) \
        .alpha_overlay(grid((10, 10), (150, 150)).fill(hex_col("#5EC6E2", 255), hex_col("#5EC6E2", 0), inflate=1.5))

    for _ in range(0):
        x, y = random.randint(50, size[0] - 50), random.randint(50, size[1] - 50)
        r = random.randint(10, 100)

        col0 = hex_col(random.choice(COLORS))[:3]
        col1 = (*col0, 1.0)
        col2 = (*col0, 0.0)

        distance = disc((x, y), r)
        shade = distance.shadow(7, 0, transparency=0.5)

        backdrop = backdrop.alpha_overlay(shade).alpha_overlay(distance.fill(col1, col2, 0))

    gradient = linear_gradient((100, 100), (900, 100), (*rand_color(), 1.0), (*rand_color(), 1.0)).transparency(0.45)
    blur = backdrop.blur(n=15, base=13)

    mask_sdf = rounded_rect((int(size[0] / 2), int(size[1] / 2)),
                            (int(size[1] / 3), int(size[1] / 3)),
                            (percent_of_min(10), percent_of_min(10), percent_of_min(10), percent_of_min(10)))

    mask = mask_sdf.generate_mask()
    outline = mask_sdf.outline((1.0, 1.0, 1.0, .25), (1.0, 1.0, 1.0, 0.0), inflate=-1.5)

    glass_col = (0.75, 0.75, 0.75, 0.75)
    glass = mask_sdf.fill(glass_col, (0.0, 0.0, 0.0, 0.0), 0)

    glass_shadow = mask_sdf.shadow(25, 0, 0.75)

    mask_sdf.fill_from_texture(gradient)

    image = backdrop.alpha_overlay(glass_shadow).mask(blur, mask).alpha_overlay(glass).alpha_overlay(
        outline).to_rgb()

    print((time.time_ns() - start) / 1e6)

    image.show()

# Example 3

context = Context((1080, 1080))
set_context(context)

scale = 0.65
offset_x = 25
offset_y = 25

inflation = 7.5

sdf = glyph_sdf("M", scale, offset_x, offset_y, path="fonts/SFUIDisplay-Bold.ttf")

bg = clear_color(hex_col("#2C2D35"))
shadow = sdf.shadow(distance=10, inflate=inflation, transparency=1.0)
colored_glyph = sdf.fill(hex_col("#e9c46a"), (0.0, 0.0, 0.0, 0.0), inflation)

bg.alpha_overlay(shadow).alpha_overlay(colored_glyph).to_rgb()  # .show()
