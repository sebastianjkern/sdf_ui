from src.sdf_ui import Context, triangle

def triangle_example():
    with Context((500, 500)) as ctx:
        p1 = (100, 200)
        p2 = (200, 200)
        p3 = (300, 300)

        tri = triangle(ctx, p1, p2, p3)
        tri.fill((1, 1, 1, 1), (0, 0, 0, 1)).show()