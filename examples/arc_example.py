from math import pi

from sdf_ui import Canvas, sdf


def arc_example():
    with Canvas((800, 500)) as ctx:
        sdf.arc((400, 250), 160, -pi * 0.15, pi * 1.15).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
            2,
        ).show(ctx)
