from sdf_ui import Canvas, sdf


def polygon_example():
    with Canvas((800, 500)) as ctx:
        sdf.polygon(((150, 100), (650, 100), (650, 180), (320, 180), (320, 320), (650, 320), (650, 400), (150, 400))).fill(
            (15, 23, 42, 255),
            (248, 250, 252, 255),
        ).show(ctx)

