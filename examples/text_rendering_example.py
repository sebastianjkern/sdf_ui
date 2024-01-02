from src.sdf_ui import Context, line

def text_rendering_example():
    with Context((500, 500)) as ctx:
        l1 = line(ctx, (250, 100), (50, 450))
        l2 = line(ctx, (250, 100), (450, 450))
        l3 = line(ctx, (100, 250), (400, 250))

        l1.union(l2) \
            .union(l3) \
            .fill(inflate=5, fg_color=(1, 1, 1, 1), bg_color=(0, 0, 0, 1)) \
            .show()