from sdf_ui import Canvas, sdf


def diamond_example():
    with Canvas((800, 500)) as ctx:
        sdf.diamond((400, 250), (170, 120)).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
        ).show(ctx)

