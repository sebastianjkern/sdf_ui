from sdf_ui import Canvas, sdf


def ellipse_example():
    with Canvas((800, 500)) as ctx:
        sdf.ellipse((400, 250), (190, 120)).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
        ).show(ctx)

