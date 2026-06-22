from sdf_ui import Canvas, sdf


def half_plane_example():
    with Canvas((800, 500)) as ctx:
        sdf.half_plane((320, 250), (1, 0)).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
        ).show(ctx)

