from sdf_ui import Canvas, sdf


def convex_polygon_example():
    with Canvas((800, 500)) as ctx:
        sdf.convex_polygon(((400, 110), (620, 250), (400, 390), (180, 250))).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
        ).show(ctx)

