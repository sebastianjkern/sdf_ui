from sdf_ui import Canvas, sdf


def ring_example():
    with Canvas((800, 500)) as ctx:
        sdf.ring((400, 250), 160, 48).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
        ).show(ctx)

