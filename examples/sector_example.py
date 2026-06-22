from sdf_ui import Canvas, sdf


def sector_example():
    with Canvas((800, 500)) as ctx:
        sdf.sector((400, 250), 170, -0.6, 1.8).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
        ).show(ctx)

