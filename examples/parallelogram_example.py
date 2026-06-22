from sdf_ui import Canvas, sdf


def parallelogram_example():
    with Canvas((800, 500)) as ctx:
        sdf.parallelogram((400, 250), (320, 180), 60).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
        ).show(ctx)

