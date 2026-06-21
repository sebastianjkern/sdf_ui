import math

from sdf_ui.anim import convert_to_video
from sdf_ui import Canvas, color, sdf
from sdf_ui.util import hex_col

def animation_example():
    with Canvas((int(1280 / 2), int(720 / 2))) as ctx:
        background = color.clear((1.0, 1.0, 1.0, 1.0)).cache("background")

        disc1 = sdf.circle((ctx.percent_x(10), ctx.percent_y(50)), ctx.percent_of_min(35)).cache("disc1")

        cache = {}
        images = []

        # Keep one cache alive for the whole frame loop so the static graph stays hot.
        for i in range(1000):
            name_of_frame = f"out/anim/{str(i).zfill(5)}.png"

            disc2 = sdf.circle(
                (ctx.percent_x(35) + ctx.percent_x(25 * math.sin(i / (math.pi * 2 * 15))), ctx.percent_y(50)),
                ctx.percent_of_min(15),
            ).uncached()

            col = hex_col("#62bb47")[:3]
            col1 = (*col, 1.0)
            col2 = (*col, 0.0)

            image = background.alpha_overlay(disc1.smooth_union(disc2).fill(col1, col2)).uncached()
            image.save(name_of_frame, ctx, cache=cache)
            images.append(name_of_frame)

        convert_to_video("out", images)
