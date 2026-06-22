from sdf_ui import Canvas, sdf


def ngon_example():
    with Canvas((800, 500)) as ctx:
        sdf.ngon((400, 250), 170, 7, rotation=0.2).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
        ).show(ctx)

