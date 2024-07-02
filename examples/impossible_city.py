from src.sdf_ui import *

from functools import reduce

TILING = 10

FG_COLOR = (0, 0, 0, 1)
BG_COLOR = (1, 1, 1, 1)
TRANSPARENT = (0, 0, 0, 0)


def isometric_cube(ctx, offset=(0, 0)):
    grid_height = ctx.percent(100/TILING)
    grid_width = ctx.percent(100/(TILING+2))

    bg = clear_color(ctx, TRANSPARENT)

    #    p7
    # p6    p4
    # p5 p2 p3
    #    p1

    p1 = (offset[0], offset[1])
    p2 = (offset[0], offset[1] + grid_height)

    p3 = (offset[0] + grid_width, offset[1] + 0.5 * grid_height)
    p4 = (offset[0] + grid_width, offset[1] + 1.5 * grid_height)

    p5 = (offset[0] - grid_width, offset[1] + 0.5 * grid_height)
    p6 = (offset[0] - grid_width, offset[1] + 1.5 * grid_height)

    p7 = (offset[0], offset[1] + 2 * grid_height)

    l1 = line(ctx, p1, p2).fill(FG_COLOR, TRANSPARENT, 2)

    l2 = line(ctx, p2, p4).fill(FG_COLOR, TRANSPARENT, 2)
    l3 = line(ctx, p1, p3).fill(FG_COLOR, TRANSPARENT, 2)

    l4 = line(ctx, p3, p4).fill(FG_COLOR, TRANSPARENT, 2)

    l5 = line(ctx, p2, p6).fill(FG_COLOR, TRANSPARENT, 2)
    l6 = line(ctx, p1, p5).fill(FG_COLOR, TRANSPARENT, 2)
    l7 = line(ctx, p5, p6).fill(FG_COLOR, TRANSPARENT, 2)
    l8 = line(ctx, p6, p7).fill(FG_COLOR, TRANSPARENT, 2)
    l9 = line(ctx, p4, p7).fill(FG_COLOR, TRANSPARENT, 2)

    return bg.alpha_overlay(l1).alpha_overlay(l2).alpha_overlay(l3).alpha_overlay(l4).alpha_overlay(l5).alpha_overlay(l6).alpha_overlay(l7).alpha_overlay(l8).alpha_overlay(l9)


def impossible_city_example():
    with Context((1080, 1080)) as ctx:
        grid_height = ctx.percent(100/TILING)
        grid_width = ctx.percent(100/(TILING+2))

        bg = clear_color(ctx, BG_COLOR)

        cubes = []

        for w in range(int((TILING+2)/2+.5)):
            for h in range(int(TILING/2+.5)):
                cubes.append(isometric_cube(ctx, offset=(grid_width + w*2*grid_width, h*2*grid_height)))

        bg.alpha_overlay(reduce(lambda x, y: x.alpha_overlay(y), cubes)).show()