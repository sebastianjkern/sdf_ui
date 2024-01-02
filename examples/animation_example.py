from src.sdf_ui import *

import math

def animation_example():
    with Context((int(1280 / 2), int(720 / 2))) as ctx:
        background = clear_color(ctx, (1.0, 1.0, 1.0, 1.0))

        disc1 = disc(ctx, (ctx.percent_x(10), ctx.percent_y(50)), ctx.percent_of_min(35))

        images = []

        for i in range(1000):
            name_of_frame = f"out/anim/{str(i).zfill(5)}.png"

            disc2 = disc(ctx, 
                (ctx.percent_x(35) + ctx.percent_x(25 * math.sin(i / (math.pi * 2 * 15))), ctx.percent_y(50)),
                ctx.percent_of_min(15))

            col = hex_col("#62bb47")[:3]
            col1 = (*col, 1.0)
            col2 = (*col, 0.0)

            image = background.alpha_overlay(disc1.smooth_union(disc2).fill(col1, col2))
            image.save(name_of_frame)
            images.append(name_of_frame)

        convert_to_video("out", images)
