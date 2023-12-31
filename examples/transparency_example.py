from src.sdf_ui import *

import random

size = (1080, 1080)

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


def transparency_example():
    with Context(size) as context:
        backdrop = radial_gradient((size[0] / 2, size[1] / 2), hex_col("#004C81", alpha=255),
                           hex_col("#062A4A", alpha=255), inner=0, outer=context.percent_of_min(75)).alpha_overlay(grid((10, 10), (50, 50)).fill(hex_col("#5EC6E2", 255), hex_col("#5EC6E2", 0), inflate=.5)).alpha_overlay(grid((10, 10), (150, 150)).fill(hex_col("#5EC6E2", 255), hex_col("#5EC6E2", 0), inflate=1.5))

        gradient = linear_gradient((100, 100), (900, 100), (*rand_color(), 1.0), (*rand_color(), 1.0)).transparency(0.45)

        blur = backdrop.blur(n=1, base=13)

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

        for i in range(5):
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

        image.show()

