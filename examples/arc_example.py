from src.sdf_ui import *

def arc_example():
    arc_width=10
    arc_inner_radius=50

    with Context((512, 512)) as context:
        d1 = disc(context, (int(512/2), int(512/2)), arc_inner_radius+arc_width)
        d2 = disc(context, (int(512/2), int(512/2)), arc_inner_radius)

        d1.subtract(d2).fill((1,1,1,1), (0, 0, 0,1)).show()
