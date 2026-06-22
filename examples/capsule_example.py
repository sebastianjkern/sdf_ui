from sdf_ui import Canvas, sdf


def capsule_example():
    with Canvas((800, 320)) as ctx:
        sdf.capsule((140, 160), (660, 160), 42).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
        ).show(ctx)

